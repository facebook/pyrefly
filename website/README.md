# Website

The website is located at https://pyrefly.org/.
It is a combination of static content (built using [Docusaurus 2](https://docusaurus.io/)) and a sandbox built with Rust/WASM.

## Generating screengrabs and compressing them

- Use Mac OS's "screencapture" to capture the screen in .mov format
- Use `ffmpeg` to convert the .mov to .mp4
  `ffmpeg -i workspace-symbols.mov -c:v libx264 -pix_fmt yuv420p -profile:v main -preset slower -an -movflags +faststart workspace-symbols.mp4`
