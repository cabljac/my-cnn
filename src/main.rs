mod lib;
use image::open;
use ndarray::{arr2, Axis};

fn main() {
    // let kernel = [-1.0f32, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    // lib::apply_convolution("pxArt.png", kernel);

    let img = open("hotdog-pixel.png").unwrap();

    let array = lib::convert_to_array(img);
    let _r = array.index_axis(Axis(2), 0).to_owned();
    let _g = array.index_axis(Axis(2), 1).to_owned();
    let _b = array.index_axis(Axis(2), 2).to_owned();

    let k = arr2(&[[-1f32, -1.0, -1.0], [-1f32, 8.0, -1.0], [-1f32, -1.0, -1.0]]);

    let conv_r = lib::convolution(_r, &k);
    let conv_g = lib::convolution(_g, &k);
    let conv_b = lib::convolution(_b, &k);

    print!("{:?}", conv_r);
    print!("{:?}", conv_g);
    print!("{:?}", conv_b);
}
