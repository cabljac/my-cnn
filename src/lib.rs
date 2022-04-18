use image::{open, DynamicImage};

pub fn apply_convolution(path: &str, kernel: Vec<f32>) -> DynamicImage {
    open(path).unwrap().filter3x3(&kernel)
}

pub fn convolution(a: ndarray::Array2<u32>, b: ndarray::Array2<u32>) -> ndarray::Array2<u32> {
    if a.shape()[0] < b.shape()[0] || a.shape()[1] < b.shape()[1] {
        panic!("input array is not big enough")
    }

    let mut c = ndarray::Array::zeros((a.shape()[0] - 2, a.shape()[1] - 2));

    for r in 0..(a.shape()[0] - 2) {
        for s in 0..(a.shape()[1] - 2) {
            c[[r, s]] = 0;
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
