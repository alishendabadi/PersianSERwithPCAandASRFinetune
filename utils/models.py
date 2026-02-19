import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import WhisperForAudioClassification
from transformers.utils import ModelOutput
from utils.multiheadattention import MultiHeadAttentivePoolinig, MultiHeadQKVAttention
import warnings
warnings.filterwarnings("ignore")


class WhisperFeatureExtractor(nn.Module):
  def __init__(self, pretrained_model, feature_layer="all"):
    super().__init__()
    self.pretrained_model = pretrained_model
    self.pretrained_model_encoder = WhisperForAudioClassification.from_pretrained(pretrained_model).encoder
    if feature_layer == "all":
      self.feature_layer = list(range(1, len(self.pretrained_model_encoder.layers)+1 ))
    elif type(feature_layer) == str : self.feature_layer = eval(feature_layer)
    else : self.feature_layer = feature_layer

    if type(self.feature_layer) == int:
      assert self.feature_layer <= len(self.pretrained_model_encoder.layers) , "REPRESENTATION LAYER IS LARGER THAN ENCODER LAYER! THIS IS NOT POSSIBLE"
    elif type(self.feature_layer) == list:
      assert max(self.feature_layer ) <= len(self.pretrained_model_encoder.layers) , "REPRESENTATION LAYER IS LARGER THAN ENCODER LAYER! THIS IS NOT POSSIBLE"
    
  def forward(self, x):
    if type(self.feature_layer) == int:
      encoder_outputs = self.pretrained_model_encoder(x, output_hidden_states=True).hidden_states
      representations = encoder_outputs[self.feature_layer - len(self.pretrained_model_encoder.layers) - 1] #output for selected transformer layers of whisper encoder
      output = {f"layer_{self.feature_layer}_representations": representations}
    elif type(self.feature_layer) == list:
      encoder_outputs = self.pretrained_model_encoder(x, output_hidden_states=True).hidden_states
      output = {f"layer_{i}_representations": encoder_outputs[i - len(self.pretrained_model_encoder.layers) - 1] for i in self.feature_layer}
    return ModelOutput(**output)
    
class MeanPoolingSER(nn.Module):
  name = "MeanPoolingSER"
  def __init__(self,
              whisper_size,
              number_of_classes
              ):
    super().__init__()
    self.projector = WhisperForAudioClassification.from_pretrained(whisper_size).projector
    self.projected_normalizer = nn.LayerNorm(self.projector.out_features)
    self.classifier = torch.nn.Linear(in_features=self.projector.out_features, out_features=number_of_classes)

  def forward(self, x):
    projected_features = self.projector(x)
    projected_features = self.projected_normalizer(projected_features)
    mean = projected_features.mean(dim=1)
    logits = self.classifier(mean)

    return ModelOutput(logits=logits)
    
class WeightedMeanPoolingSER (nn.Module):
  def __init__(self,
               feature_extractor_model,
               number_of_classes,
               model_embedding_dim=256,
               return_attention_weights=False):
    super().__init__()
    self.return_weights = return_attention_weights
    self.model_embedding_dim = model_embedding_dim
    self.feature_extractor = feature_extractor_model
    self.projector = WhisperForAudioClassification.from_pretrained(feature_extractor_model.pretrained_model).projector
    self.layernorm = nn.LayerNorm(model_embedding_dim)
    self.attention_scorer = nn.Linear(in_features=model_embedding_dim, out_features=1, bias=True)
    self.classifier = torch.nn.Linear(in_features=model_embedding_dim, out_features=number_of_classes)

  def forward(self, x):
    features = self.feature_extractor(x).acoustic_features
    projected = self.projector(features)
    out = F.leaky_relu(projected)
    out = self.layernorm(out)
    attention_scores = self.attention_scorer(out)
    attention_scores = F.relu(attention_scores)
    attention_weights = F.softmax(attention_scores, dim=1)
    weighted_output = out*attention_weights
    weighted_mean_pooled_output = weighted_output.sum(dim=1)
    logits = self.classifier(weighted_mean_pooled_output)

    if self.return_weights:
      return ModelOutput(logits=logits, attention_weights=attention_weights)
    else:
      return ModelOutput(logits=logits)
    
class MultiHeadQnoGradSER(nn.Module):
  def __init__(self,
              feature_extractor_model,
              number_of_classes,
              heads,
              mh_hidden_dim_sum,
              dropout=0.5,
              model_embedding_dim=256,
              return_attention_weights=False,
              skip_connection=True):
    super().__init__()
    self.skip_connection = skip_connection
    self.return_weights = return_attention_weights
    self.model_embedding = model_embedding_dim
    self.feature_extractor = feature_extractor_model
    self.projector = WhisperForAudioClassification.from_pretrained(feature_extractor_model.pretrained_model).projector
    self.projected_normalizer = nn.LayerNorm(model_embedding_dim)
    self.query_parameter = nn.Parameter(torch.rand(1, mh_hidden_dim_sum))
    self.multihead = nn.MultiheadAttention(embed_dim=mh_hidden_dim_sum,
                                           kdim=model_embedding_dim,
                                           vdim=model_embedding_dim,
                                           num_heads=heads,
                                           batch_first=True,
                                           dropout=dropout)
    self.dimentin_fix = nn.Linear(in_features=mh_hidden_dim_sum, out_features=model_embedding_dim)
    self.mh_normalizer = nn.LayerNorm(model_embedding_dim)
    self.classifier = torch.nn.Linear(in_features=model_embedding_dim, out_features=number_of_classes)

  def forward(self, x):
    features = self.feature_extractor(x).acoustic_features
    projected_features = self.projector(features)
    projected_features = F.leaky_relu(projected_features)
    projected_features = self.projected_normalizer(projected_features)
    stacked_parameter = self.query_parameter.unsqueeze(0).repeat(x.size()[0], 1, 1)
    if not self.skip_connection:
      weighted_output, attention_output_weights= self.multihead(query=stacked_parameter , key=projected_features, value=projected_features)
      fixed_dimention = self.dimentin_fix(weighted_output)
      fixed_dimention = torch.squeeze(fixed_dimention, dim=1)
      out = self.mh_normalizer(fixed_dimention)
    elif self.skip_connection:
      skip_connection = torch.mean(projected_features, dim=-2)
      weighted_output, attention_output_weights= self.multihead(query=stacked_parameter , key=projected_features, value=projected_features)
      fixed_dimention = self.dimentin_fix(weighted_output)
      fixed_dimention = torch.squeeze(fixed_dimention, dim=1)
      added = (skip_connection + fixed_dimention)/2
      out = self.mh_normalizer(added)
    logits = self.classifier(out)

    if self.return_weights:
      return ModelOutput(logits=logits, attention_weights=attention_output_weights)
    else:
      return ModelOutput(logits=logits)
    
class QKVMultiheadPoolingSER(nn.Module):
  name = "QKVAttentivePoolingSER"
  def __init__(self,
              number_of_classes,
              whisper_size,
              d_model,
              n_heads,
              dropout,
              hidden_dim_each_head,
              return_attention_weights=False):
    super().__init__()
    self.return_weights = return_attention_weights
    self.d_model = d_model
    self.projector = WhisperForAudioClassification.from_pretrained(whisper_size).projector
    self.projected_normalizer = nn.LayerNorm(d_model)
    self.multihead = MultiHeadQKVAttention(d_k= hidden_dim_each_head, d_model=d_model, n_heads=n_heads, dropout=dropout)
    self.mh_normalizer = nn.LayerNorm(d_model)
    self.classifier = torch.nn.Linear(in_features=d_model, out_features=number_of_classes)

  def forward(self, x):
    projected_features = self.projector(x)
    projected_features = self.projected_normalizer(projected_features)
    global_mean = torch.mean(input=projected_features, dim=-2, keepdim=True)
    attention_output, attention_weights = self.multihead(q=global_mean, k=projected_features, v=projected_features)
    normed = self.mh_normalizer(attention_output)
    logits = self.classifier(normed)

    if self.return_weights:
      return ModelOutput(logits=logits, attention_weights=attention_weights)
    else:
      return ModelOutput(logits=logits)


class MultiHeadAttentivePoolingSER(nn.Module):
  name = "MultiHeadAttentivePoolingSER"
  def __init__(self,
              number_of_classes,
              whisper_size,
              d_model,
              n_heads,
              dropout,
              hidden_dim_each_head,
              return_attention_weights=False):
    super().__init__()
    self.return_weights = return_attention_weights
    self.d_model = d_model
    self.projector = WhisperForAudioClassification.from_pretrained(whisper_size).projector
    self.projected_normalizer = nn.LayerNorm(d_model)
    self.multihead = MultiHeadAttentivePoolinig(d_attention=hidden_dim_each_head, d_model=d_model, n_heads=n_heads, dropout=dropout)
    self.mh_normalizer = nn.LayerNorm(d_model)
    self.classifier = torch.nn.Linear(in_features=d_model, out_features=number_of_classes)

  def forward(self, x):
    projected_features = self.projector(x)
    projected_features = self.projected_normalizer(projected_features)
    attention_output, attention_weights = self.multihead(k=projected_features, v=projected_features)
    normed = self.mh_normalizer(attention_output)
    logits = self.classifier(normed)

    if self.return_weights:
      return ModelOutput(logits=logits, attention_weights=attention_weights)
    else:
      return ModelOutput(logits=logits)