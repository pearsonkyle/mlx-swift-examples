#  Stable Diffusion

Stable Diffusion in MLX. The implementation was ported from Hugging Face's
[diffusers](https://huggingface.co/docs/diffusers/index) and 
[mlx-examples/stable_diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
Model weights are downloaded directly from the Hugging Face hub. The implementation currently
supports the following models:

- [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)
- [stabilitiai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

Additionally, **custom SDXL models** can be loaded from local `.safetensors` checkpoint files
(e.g., models from CivitAI), with support for:
- **Separate VAE** loading (e.g., `sdxl.vae.safetensors`)
- **LoRA** weight loading and fusion
- **360° panorama generation** with seamless X-axis tiling

## Usage

See [StableDiffusionExample](../../Applications/StableDiffusionExample) and
[image-tool](../../Tools/image-tool) for examples of using this code.

### Standard Usage (SDXL Turbo)

The basic sequence is:

- download & load the model
- generate latents
- evaluate the latents one by one
- decode the last latent generated
- you have an image!

```swift
let configuration = StableDiffusionConfiguration.presetSDXLTurbo

let generator = try configuration.textToImageGenerator(
    configuration: model.loadConfiguration)

generator.ensureLoaded()

let parameters = generate.evaluateParameters(configuration: configuration)
let latents = generator.generateLatents(parameters: parameters)

var lastXt: MLXArray?
for xt in latents {
    eval(xt)
    lastXt = xt
}

if let lastXt {
    var raster = decoder(lastXt[0])
    raster = (raster * 255).asType(.uint8).squeezed()
    eval(raster)
    try Image(raster).save(url: url)
}
```

### Custom Model + LoRA + Panorama

Load a custom SDXL checkpoint from a `.safetensors` file with optional VAE and LoRA:

```swift
// 1. First download SDXL Turbo (provides tokenizer/scheduler files shared across all SDXL models)
let sdxlTurbo = StableDiffusionConfiguration.presetSDXLTurbo
try await sdxlTurbo.download()

// 2. Load custom checkpoint
let sd = try loadStableDiffusionXLFromSingleFile(
    url: URL(filePath: "/path/to/dreamshaperXL.safetensors"),
    vaeUrl: URL(filePath: "/path/to/sdxl.vae.safetensors"),
    dType: .float16
)

// 3. Load and fuse LoRA weights
try loadAndFuseLora(sd, loraUrl: URL(filePath: "/path/to/360Redmond.safetensors"), scale: 1.0)

// 4. Create panorama generator
let generator = PanoramaGenerator(sd, width: 2048, height: 1024)
generator.ensureLoaded()

// 5. Generate panorama
let parameters = PanoramaParameters(
    prompt: "Glowing mushrooms around pyramids, equirectangular, 360 panorama, cinematic",
    steps: 8,
    cfgScale: 3.0
)

let decoder = generator.detachedDecoder()
let latents = generator.generateLatents(parameters: parameters)

var lastXt: MLXArray?
for xt in latents {
    eval(xt)
    lastXt = xt
}

if let lastXt {
    let decoded = decoder(lastXt)
    let raster = (decoded * 255).asType(.uint8).squeezed()
    try Image(raster).save(url: URL(filePath: "/tmp/panorama_360.png"))
}
```

### Required Model Files

For custom model loading, you need:

1. **SDXL Turbo** — Downloaded once for tokenizer and scheduler config files (shared architecture)
2. **Checkpoint** — A `.safetensors` file (e.g., from CivitAI)
   - [DreamShaper XL Lightning](https://civitai.com/api/download/models/354657)
3. **VAE** (optional) — A separate `.safetensors` VAE file
   - [SDXL VAE FP16 Fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
4. **LoRA** (optional) — A `.safetensors` LoRA file
   - [360° Redmond](https://civitai.com/api/download/models/143197)
