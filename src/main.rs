mod lib;

fn main() {
    let image_array = lib::get_image_array("pxArt.png");

    println!("{}", image_array.slice(ndarray::s![.., .., 2]));
}
