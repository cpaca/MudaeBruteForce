

typedef std::uint16_t setSize_t;

template <typename T>
/**
 * Gets the maximum value for an unsigned value. Uses the method described in https://stackoverflow.com/a/39878362
 */
T get_unsigned_max(){
    return ~(static_cast<T>(0));
}