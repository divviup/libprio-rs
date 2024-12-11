#!/usr/bin/env sage

# This file recomputes the values in src/fp.rs for each FFT-friendly finite
# field.

import pprint


class Field:
    # The name of the field.
    name: str

    # The prime modulus that defines the field.
    modulus: Integer

    # A generator element that generates a large subgroup with an order that's
    # a power of two. This is _not_ in Montgomery representation.
    generator_element: Integer

    # The base 2 logarithm of the order of the FFT-friendly multiplicative
    # subgroup. The generator element will be a 2^num_roots-th root of unity.
    num_roots: Integer

    # The base used for multiprecision arithmetic. This should be a power of
    # two, and it should ideally be the machine word size of the target
    # architecture.
    r: Integer

    # The radix. This must be a multiple of the base "r", and must be coprime
    # to the prime "p".
    R: Integer

    def __init__(self, name, modulus, generator_element, r, R):
        assert is_prime(modulus)
        assert R % r == 0
        assert R % modulus != 0
        assert modulus % R != 0

        self.name = name
        self.modulus = modulus
        self.generator_element = generator_element
        self.r = r
        self.R = R

        self.num_roots = None
        for (prime, power) in factor(modulus - 1):
            if prime == 2:
                self.num_roots = power
                break
        else:
            raise Exception(
                "Multiplicative subgroup order is not a multiple of two"
            )

    def mu(self):
        """
        Computes mu, a constant used during multiplication. It is defined by
        mu = (-p)^-1 mod r, where r is the modulus implicitly used in wrapping
        machine word operations.
        """
        return (-self.modulus).inverse_mod(self.r)

    def r2(self):
        """
        Computes R^2 mod p. This constant is used when converting into
        Montgomery representation. R is the machine word-friendly modulus
        used in the Montgomery representation.
        """
        return self.R ^ 2 % self.modulus

    def to_montgomery(self, element):
        """
        Transforms an element into its Montgomery representation.
        """

        return element * self.R % self.modulus

    def bit_mask(self):
        """
        An integer with the same bit length as the prime modulus, but with all
        bits set.
        """
        return 2 ^ (self.modulus.nbits()) - 1

    def roots(self):
        """
        Returns a list of roots of unity, in Montgomery representation. The
        value at index i is a 2^i-th root of unity. Note that the first array
        element will thus be the Montgomery representation of one.
        """
        return [
            self.to_montgomery(
                pow(
                    self.generator_element,
                    2 ^ (self.num_roots - i),
                    self.modulus,
                )
            )
            for i in range(min(self.num_roots, 20) + 1)
        ]

    def log2_base(self):
        """
        Returns log2(r), where r is the base used for multiprecision arithmetic.
        """
        return log(self.r, 2)

    def log2_radix(self):
        """
        Returns log2(R), where R is the machine word-friendly modulus
        used in the Montgomery representation.
        """
        return log(self.R, 2)


FIELDS = [
    Field(
        "FieldPrio2, u32",
        2 ^ 20 * 4095 + 1,
        3925978153,
        2 ^ 32,
        2 ^ 32,
    ),
    Field(
        "Field64, u64",
        2 ^ 32 * 4294967295 + 1,
        pow(7, 4294967295, 2 ^ 32 * 4294967295 + 1),
        2 ^ 64,
        2 ^ 64,
    ),
    Field(
        "Field128, u128",
        2 ^ 66 * 4611686018427387897 + 1,
        pow(7, 4611686018427387897, 2 ^ 66 * 4611686018427387897 + 1),
        2 ^ 64,
        2 ^ 128,
    ),
]
for field in FIELDS:
    print(field.name)
    print(f"p: {field.modulus}")
    print(f"mu: {field.mu()}")
    print(f"r2: {field.r2()}")
    print(f"g: {field.to_montgomery(field.generator_element)}")
    print(f"num_roots: {field.num_roots}")
    print(f"bit_mask: {field.bit_mask()}")
    print("roots:")
    pprint.pprint(field.roots())
    print(f"log2_base: {field.log2_base()}")
    print(f"log2_radix: {field.log2_radix()}")
    print()
