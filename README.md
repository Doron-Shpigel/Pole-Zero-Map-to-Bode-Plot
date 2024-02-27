# Pole-Zero Map GUI

This code provides a simple graphical user interface (GUI) for manipulating poles and zeros in a pole-zero map, allowing you to observe their impact on the Bode plot.

### Constants:
- `drawArea`: 10
- `GAIN`: 100
- `nDoublePole`: 1
- `nDoubleZero`: 1
- `nSinglePole`: 1
- `nSingleZero`: 1

You can customize the number of poles or zeros to interact with. The `drawArea` parameter defines the "sandbox" of the map. Placing a pole or zero inside this sandbox affects the transfer function and consequently alters the Bode plot.

The `GAIN` constant acts as a multiplier for the transfer function.

Poles are colored red, and Zeros are colored blue. Real Poles or Zeros are restricted to movement along the real axis, taking the shape of a wedge from a circle. Complex conjugate pairs have a circular shape, where only the first point within the pair can be moved. The second point of the pair is its conjugate and is represented with a brighter color (pink or cyan).
Poles are filled with color, while Zeros are represented with a hole instead of fill, and only the boundaries of the shape are visible.

Enjoy experimenting with the GUI! :)
