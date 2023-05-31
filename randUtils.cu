/**
 * Generates a random value, then updates the seed.
 * This is statistical randomness, not cryptographic randomness.
 * Also note that this method simply returns the seed; this is done because it makes some shorthand things easier
 * (i.e. you can do generateRandom(seed)%limit instead of having seed = generateRandom(seed); num = seed%limit)
 * @param seed The "seed" of the randomness.
 * @return A random number from 0 to 2^64-1
 */
__device__ size_t generateRandom(size_t &seed){
    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    // These are the values that newlib uses, and while there is a note saying that not all of the values are ideal
    // this is one of the only ones that uses modulus 2^64
    seed = (6364136223846793005*seed) + 1;
    return seed;
}

/**
 * Generates a random value (with generateRandom) that is less than the maxVal specified.
 * This is more computationally efficient than generateRandom(seed) % maxVal
 * @param seed The "seed" of the randomness.
 * @return A random number from 0 to maxVal-1
 */
__device__ size_t generateRandom(size_t &seed, size_t maxVal){
    // note that andOp's binary representation is all 0s everywhere to the left of maxVal's most significant "1" bit
    // and 1s everywhere to the right of and including maxVal's most significant "1" bit.
    size_t andOp = maxVal;
    andOp |= andOp >> 1;
    andOp |= andOp >> 2;
    andOp |= andOp >> 4;
    andOp |= andOp >> 8;
    andOp |= andOp >> 16;
    andOp |= andOp >> 32;

    while(true){
        size_t randNum = generateRandom(seed);
        // Apparantly the less-significant bits repeat "more quickly" so this is a way of getting "more randomness"
        // It shouldn't really matter, but we're dealing with a million checks per second... it's best to do this just in case.
        randNum ^= randNum >> 32;
        // Fast operation to get a number that is *most likely* (though not guaranteed) to be <= maxVal
        randNum &= andOp;
        // Since randNum is not guaranteed to be <= maxVal, only return randNum if it is.
        if(randNum < maxVal){
            return randNum;
        }
    }
}