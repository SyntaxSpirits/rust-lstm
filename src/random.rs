//! Random number generation for weight initialisation, dropout, zoneout and sampling.

use ndarray::{Array2, Ix2, ShapeBuilder};
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

/// Reseeds the generator of the current thread.
///
/// After `seed(s)`, weight initialisation, dropout and zoneout masks and text sampling
/// on this thread produce the same values on every run. Each thread has its own
/// generator, seeded from the operating system until `seed` is called.
pub fn seed(seed: u64) {
    RNG.with(|rng| *rng.borrow_mut() = StdRng::seed_from_u64(seed));
}

pub(crate) fn with_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    RNG.with(|rng| f(&mut rng.borrow_mut()))
}

pub(crate) fn random_array<Sh, D>(shape: Sh, dist: D) -> Array2<f64>
where
    Sh: ShapeBuilder<Dim = Ix2>,
    D: Distribution<f64>,
{
    with_rng(|rng| Array2::random_using(shape, dist, rng))
}

/// A sample from U[0, 1).
pub(crate) fn unit() -> f64 {
    with_rng(|rng| rng.gen())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn same_seed_gives_same_values() {
        seed(7);
        let a = random_array((3, 4), Uniform::new(0.0, 1.0));
        seed(7);
        let b = random_array((3, 4), Uniform::new(0.0, 1.0));
        seed(8);
        let c = random_array((3, 4), Uniform::new(0.0, 1.0));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
