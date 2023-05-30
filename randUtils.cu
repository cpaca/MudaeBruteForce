/**
 * Generates a random value, then updates the seed.
 * This is statistical randomness, not cryptographic randomness.
 * Also note that this method simply returns the seed; this is done because it makes some shorthand things easier
 * (i.e. you can do generateRandom(seed)%limit instead of having seed = generateRandom(seed); num = seed%limit)
 * @param seed The "seed" of the randomness.
 * @return
 */
__device__ size_t generateRandom(size_t &seed){
    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    // These are the values that newlib uses, and while there is a note saying that not all of the values are ideal
    // this is one of the only ones that uses modulus 2^64
    seed = (6364136223846793005*seed) + 1;
    return seed;
}