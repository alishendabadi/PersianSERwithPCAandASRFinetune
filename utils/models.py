import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import WhisperForAudioClassification
from transformers.utils import ModelOutput
from utils.multiheadattention import MultiHeadAttentivePoolinig, MultiHeadQKVAttention
import warnings
warnings.filterwarnings("ignore")

class PCAProjector(nn.Module):
  """
  Parameter-free dimensionality reduction via PCA.
  Fitted once per fold on the training split only, stored as frozen buffers.
  Replaces the Whisper projector to avoid overfitting on small datasets.
  """
  def __init__(self, n_components: int):
    super().__init__()
    self.n_components = n_components
    # None buffers are valid in PyTorch; replaced by fit() before use
    self.register_buffer('components_', None)  # [n_components, input_dim]
    self.register_buffer('mean_', None)         # [input_dim]

  @property
  def out_features(self):
    return self.n_components

  def fit(self, X: torch.Tensor):
    """
    Fit PCA on training-fold representations.
    X: float tensor of shape [N_samples * T, input_dim]
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=self.n_components)
    pca.fit(X.cpu().float().numpy())
    components = torch.tensor(pca.components_, dtype=torch.float32)  # [n_components, input_dim]
    mean_ = torch.tensor(pca.mean_, dtype=torch.float32)             # [input_dim]
    self.register_buffer('components_', components)
    self.register_buffer('mean_', mean_)
    self.explained_variance_sum = float(pca.explained_variance_ratio_.sum())
    print(f"  PCAProjector fitted: {mean_.shape[0]}D → {self.n_components}D "
          f"(cumulative explained variance: {self.explained_variance_sum:.3f})")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: [B, T, input_dim] → [B, T, n_components]"""
    assert self.components_ is not None, \
      "PCAProjector has not been fitted yet. Call model.projector.fit(data) before training."
    return (x - self.mean_) @ self.components_.T

class WhisperFeatureExtractor(nn.Module):
  def __init__(self, pretrained_model, feature_layer="all"):
    super().__init__()
    self.pretrained_model = pretrained_model
    self.pretrained_model_encoder = WhisperForAudioClassification.from_pretrained(pretrained_model).encoder.float()
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
    
class QKVMultiheadPoolingSER(nn.Module):
  name = "QKVAttentivePoolingSER"
  def __init__(self,
              number_of_classes,
              whisper_size,
              d_model,
              n_heads,
              dropout,
              hidden_dim_each_head,
              use_pca=False,
              return_attention_weights=False):
    super().__init__()
    self.return_weights = return_attention_weights
    self.d_model = d_model
    if use_pca:
      self.projector = PCAProjector(n_components=d_model)
    else:
      self.projector = WhisperForAudioClassification.from_pretrained(whisper_size).projector.float()
    self.projected_normalizer = nn.LayerNorm(d_model)
    self.multihead = MultiHeadQKVAttention(d_k= hidden_dim_each_head, d_model=d_model, n_heads=n_heads, dropout=dropout)
    self.mh_normalizer = nn.LayerNorm(d_model)
    self.classifier = torch.nn.Linear(in_features=d_model, out_features=number_of_classes).float()

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
              use_pca=False,
              return_attention_weights=False):
    super().__init__()
    self.return_weights = return_attention_weights
    self.d_model = d_model
    if use_pca:
      self.projector = PCAProjector(n_components=d_model)
    else:
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
