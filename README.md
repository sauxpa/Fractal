# Fractal
Homemade fractal art, powered by Python (meh...) and Numba (with JIT and multithread/GPU!).

## To generate .gif or .mp4 of rotating Julia sets
In a terminal, run :
python julia_anime.py -f your_file_name.gif

## To generate .gif or .mp4 of zooming Julia sets
python julia_zoom.py -f your_file_name.gif

## To generate .gif or .mp4 of 2d projection of quaternion Julia sets
In a terminal, run :
python julia_quaternion_print.py -f your_file_name.gif

All these functions support many more arguments, see source or command help.

# List of available fractals

##  Burning ship
Variant of Mandelbrot set with absolute value update function: https://en.wikipedia.org/wiki/Burning_Ship_fractal.

<img src="./burning_ship/images/burning_ship.png"
     style="float: left; margin-right: 10px;" />
     
## Chaos game
Iterative fractal deformation of an initial shape: https://en.wikipedia.org/wiki/Chaos_game.


<img src="./chaos_game/sierpinski_batman.gif"
     style="float: left; margin-right: 10px;" />
     

<img src="./chaos_game/sierpinski_dachshund.gif"
     style="float: left; margin-right: 10px;" />
     
## Diffeobrot set
Continuous time analogue of the Mandelbrot equation:
dz(t)/dt + z(t) = z(t-t0)^2 + c, with z(0) = 0.

The diffeobrot set is the set of point c in the complex plane such that this system is stable. This varies with the delay parameter t0, more precisely the latter controls the "smoothness" of the resulting Mandelbrot-like set.

## Julia set 
Dual sets of the Mandelbrot set: https://en.wikipedia.org/wiki/Julia_set.

<img src="./julia/images/julia_0.285_0.01.png"
     style="float: left; margin-right: 10px;" />
     
## Julia quaternion
Same definition as complex Julia sets, only applied with quaternion algebra. Visualization is projected from 4d to 3d via slicing, then displayed as a 2d moving shape.

<img src="./julia_quaternion/images/julia_quaternion_0285_001_0005_0005.gif"
     style="float: left; margin-right: 10px;" />
     
## Magnetbrot
Yet another variation on Mandelbrot, this time with a rational update function rather than a polynomial one.
     
<img src="./magnetbrot/pictures/magnetbrot.png"
     style="float: left; margin-right: 10px;" />
     
## Mandelbrot
Dual set of Julia sets: https://en.wikipedia.org/wiki/Mandelbrot_set.

<img src="./mandelbrot/images/mandelbrot.png"
     style="float: left; margin-right: 10px;" />
