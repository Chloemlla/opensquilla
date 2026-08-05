# OpenSquilla App Icons

This directory holds the bundled application icons for the Tauri v2 desktop
shell. The Tauri bundler references these paths from `tauri.conf.json` under
`bundle.icon`, `bundle.windows.nsis.installerIcon`, `bundle.windows.nsis.*Image`,
and `app.trayIcon.iconPath`.

## Required icon files

| File | Platform | Purpose |
| --- | --- | --- |
| `32x32.png` | Linux / fallback | Small PNG used by the bundle icon set and the Linux desktop entry. |
| `128x128.png` | Linux / fallback | Standard app icon PNG. |
| `128x128@2x.png` | Linux / fallback | HiDPI 256x256 PNG (retina-style naming for the bundle icon set). |
| `icon.png` | All (tray) | Tray icon and general-purpose 512x512+ source PNG. |
| `icon.ico` | Windows | Multi-resolution Windows icon (16, 24, 32, 48, 64, 128, 256). Used for `.exe`/NSIS installer. |
| `icon.icns` | macOS | Apple icon set for `.app` and `.dmg` bundles. |
| `nsis-sidebar.bmp` | Windows (NSIS) | 164x314 24-bit BMP installer sidebar image. |
| `nsis-header.bmp` | Windows (NSIS) | 150x57 24-bit BMP installer header image. |

> The two NSIS BMP images are only required for the Windows NSIS installer
> branding; the bundler will still produce a working installer without them.

## Generating icons from a source image

Do not hand-edit the binary icon formats. Generate the full set from a single
high-resolution source PNG (at least 1024x1024, square, with transparent or
solid background) using the Tauri CLI:

```sh
# Install the Tauri CLI if you don't have it (Rust toolchain required):
cargo install tauri-cli --version "^2"

# Generate every Tauri icon format from a source PNG:
cargo tauri icon ./icon-source.png

# Or from the default source location (./app-icon.png):
cargo tauri icon
```

`cargo tauri icon` writes the standard set (`32x32.png`, `128x128.png`,
`128x128@2x.png`, `icon.icns`, `icon.ico`, plus iOS/Android sizes) into the
`src-tauri/icons/` directory. It derives the Windows `.ico` and macOS `.icns`
from the source PNG automatically.

### NSIS installer branding images

`cargo tauri icon` does not produce the NSIS sidebar/header BMPs. Generate
them from the source PNG with ImageMagick (24-bit BMP, no alpha):

```sh
magick ./icon-source.png -resize 164x314^ -gravity center -extent 164x314 \
  -background white -alpha off -type TrueColor src-tauri/icons/nsis-sidebar.bmp

magick ./icon-source.png -resize 150x57^ -gravity center -extent 150x57 \
  -background white -alpha off -type TrueColor src-tauri/icons/nsis-header.bmp
```

### Updating the tray icon

The tray icon path (`icons/icon.png` in `app.trayIcon.iconPath`) should be a
square PNG (32x32 or 64x64 looks crispest in the menu bar / system tray).
Replace `icon.png` and re-run `cargo tauri icon` to keep the bundle icons in
sync with the tray.

## Source art

Keep the original source artwork (SVG/PNG) outside this directory — only the
generated output belongs in `src-tauri/icons/`. The committed `icon.png`,
`32x32.png`, and `icon.ico` are checked into git as the default set; regenerate
them whenever the brand mark changes.
