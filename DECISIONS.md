# NaviSpill AI - Project Decisions Record

## Decision 1: Execution Hardware Strategy (CPU Optimization)
- **Context**: CUDA detection confirmed `torch.cuda.is_available() == False`. The local environment is CPU-only.
- **Decision**: Optimize model architectures and batch sizes for CPU execution. Use ResNet34 with lightweight decoder channels, train with input tile dimensions of $256 \times 256$ pixels and batch size $B=4$. Keep demonstration fine-tuning epochs to 1–3 epochs with early stopping to ensure responsive completion within the 2–3 hour constraint.
- **Consequence**: Training finishes quickly without hanging the workstation, and inference latency remains responsive (<1.5 seconds per tile).

## Decision 2: SAR Single-Channel Pretrained Encoder Adaptation
- **Context**: Standard ImageNet pretrained backbones expect 3 RGB channels ($C=3$). Sentinel-1 SAR imagery is single-channel ($C=1$, VV or VH polarization backscatter).
- **Decision**: Adapt the first convolution layer (`conv1`) of ResNet34 by averaging the 3 RGB weight filters into a single channel:
  $$W_{SAR} = \frac{1}{3} (W_R + W_G + W_B)$$
  This retains the edge, texture, and gradient feature extractors learned from ImageNet while accepting 1-channel grayscale SAR arrays directly.
- **Consequence**: Full leverage of pretrained ImageNet weights with zero channel mismatch errors and faster convergence.

## Decision 3: Loss Function Formulation
- **Context**: Marine oil spills represent high class imbalance (background water covers >95% of pixels, oil slick occupies <5%). Standard Cross-Entropy produces trivial all-background predictions.
- **Decision**: Use a hybrid loss combining `BCEWithLogitsLoss` and `DiceLoss`:
  $$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{BCE} + \beta \cdot \mathcal{L}_{Dice}$$
  Defaulting to $\alpha = 0.5, \beta = 0.5$.
- **Consequence**: Dice loss directly penalizes boundary misalignments and false negatives in tiny spill clusters, preventing the model from collapsing to background-only output.

## Decision 4: Metric CRS Reprojection for Attribution
- **Context**: Satellite imagery and AIS positions are typically recorded in geographic coordinates (EPSG:4326, latitude/longitude in degrees). Calculating distances in degrees introduces severe distortion at non-equatorial latitudes.
- **Decision**: Reproject polygons and vessel positions into a local metric Coordinate Reference System (e.g. UTM zone or EPSG:3857 Web Mercator) using PyProj / GeoPandas prior to computing Euclidean distances, intersection, and buffer rings.
- **Consequence**: Vessel distance metrics are measured in true SI meters/kilometers, maintaining scientific and engineering accuracy.

## Decision 5: Non-Judicial Attribution Scoring Disclaimer
- **Context**: Determining vessel responsibility for marine pollution involves legal, operational, and ocean current modeling factors beyond basic spatial proximity.
- **Decision**: Implement a transparent, weighted attribution scoring model (spatial distance, temporal proximity, trajectory alignment, speed plausibility) and explicitly display a clear investigative warning: *"Attribution score is an investigative prioritization aid based on spatio-temporal proximity and trajectory analysis, not conclusive judicial proof of liability."*
- **Consequence**: Academic integrity and ethical AI standards are strictly upheld.
