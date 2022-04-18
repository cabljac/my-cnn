use image::{open, DynamicImage};
use ndarray::Array3;

pub fn convert_to_array(img: DynamicImage) -> Array3<f32> {
    let width = usize::try_from(img.width()).unwrap();
    let height = usize::try_from(img.height()).unwrap();

    let mut array = Array3::<f32>::zeros((width, height, 3usize));

    for pixel in img.to_rgb32f().enumerate_pixels() {
        let x = usize::try_from(pixel.0).unwrap();
        let y = usize::try_from(pixel.1).unwrap();
        let rgb = pixel.2;
        array[[x, y, 0]] = rgb[0];
        array[[x, y, 1]] = rgb[1];
        array[[x, y, 2]] = rgb[2];
    }
    array
}

pub fn apply_convolution(path: &str, kernel: [f32; 9]) -> DynamicImage {
    open(path).unwrap().filter3x3(&kernel)
}

pub fn convolution(a: ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    if a.shape()[0] < b.shape()[0] || a.shape()[1] < b.shape()[1] {
        panic!("input array is not big enough")
    }

    let mut c = ndarray::Array2::<f32>::zeros((
        1 + a.shape()[0] - b.shape()[0],
        1 + a.shape()[1] - b.shape()[1],
    ));

    // print!("{:?} \n \n {:?} \n \n", a, b);
    for r in 0..c.shape()[0] {
        for s in 0..c.shape()[1] {
            for i in 0..b.shape()[0] {
                for j in 0..b.shape()[1] {
                    c[[r, s]] += a[[r + i, s + j]] * b[[i, j]];
                }
            }
        }
    }
    c
}

#[test]
fn convolution_test() {
    let a = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]);

    let b = ndarray::arr2(&[[1, 0, 1], [0, 1, 0], [1, 0, 1]]);

    let c = ndarray::arr2(&[[25], [20]]);

    assert_eq!(convolution(a, b), c);
}
