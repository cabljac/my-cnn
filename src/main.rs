mod lib;
use image::{open, GrayImage, Luma};
use ndarray::{arr2, Array2, Axis};

fn main() {
    // let kernel = [-1.0f32, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    // lib::apply_convolution("pxArt.png", kernel);

    let img = open("hotdogman.jpg").unwrap();

    let array = lib::convert_to_array(img);
    let _r = array.index_axis(Axis(2), 0).to_owned();
    let _g = array.index_axis(Axis(2), 1).to_owned();
    let _b = array.index_axis(Axis(2), 2).to_owned();

    let k = arr2(&[[-1f32, -1.0, -1.0], [-1f32, 8.0, -1.0], [-1f32, -1.0, -1.0]]);

    let conv_r = lib::convolution(_r, &k);
    let conv_g = lib::convolution(_g, &k);
    let conv_b = lib::convolution(_b, &k);

    // print!("{:?}", conv_r);
    // print!("{:?}", conv_g);
    // print!("{:?}", conv_b);

    let sum = conv_r + conv_g + conv_b;

    let min = sum.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    let normalized = sum.map(|x| x - min);
    let max = normalized.iter().fold(-1.0f32, |a, &b| a.max(b));
    let normalized_2 = normalized.map(|x| (255f32 * (x / max)) as u8);
    print!("{:?}", sum);
    print!("{:?}", normalized_2);
    // let buffer = GrayImage::from_raw(
    //     sum.shape()[0] as u32,
    //     sum.shape()[1] as u32,
    //     sum.into_raw_vec(),
    // );
    let width = normalized_2.shape()[0] as u32;
    let height = normalized_2.shape()[1] as u32;

    let mut img = GrayImage::new(width, height);

    for x in 0..width - 1 {
        for y in 0..height - 1 {
            let val = normalized_2[[x as usize, y as usize]];
            img.put_pixel(x, y, Luma([val]));
        }
    }
    img.save("test.png").unwrap();
}
