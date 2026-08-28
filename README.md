# playground

Small public experiments that are safe to publish.

## Particle Type Lab

`index.html` is a dependency-free browser demo inspired by a particle/3D typography experiment. It rasterizes Japanese glyphs, builds a distance field with the Felzenszwalb/Huttenlocher transform, uses that field to create a beveled particle volume, then animates the particles between a scattered floor state and 3D letterforms.

- No external JavaScript or font dependency
- Canvas 2D rendering with pseudo-3D projection and depth sorting
- Touch / pointer orbit controls
- Adaptive particle count for mobile
- Deployed with GitHub Pages through GitHub Actions

Open the Pages site after the deployment workflow finishes.
