
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field(u32);

pub const MODULUS: u32 = 4293918721; // 2^32 - 2^20 + 1 - a prime
pub const GENERATOR: u32 = 3925978153; // generator for the multiplicative subgroup
pub const N_ROOTS: u32 = 1 << 20; // number of primitive roots

impl std::ops::Add for Field {
    type Output = Field;

    fn add(self, rhs: Self) -> Self {
        self - Field(MODULUS - rhs.0)
    }
}

impl std::ops::Sub for Field {
    type Output = Field;

    fn sub(self, rhs: Self) -> Self {
        let l = self.0;
        let r = rhs.0;

        if l >= r {
            Field(l - r)
        } else {
            Field(MODULUS - r + l)
        }
    }
}

impl std::ops::Mul for Field {
    type Output = Field;

    fn mul(self, rhs: Self) -> Self {
        let l = self.0 as u64;
        let r = rhs.0 as u64;
        let mul = l * r;
        Field((mul % (MODULUS as u64)) as u32)
    }
}

impl std::ops::Div for Field {
    type Output = Field;

    fn div(self, rhs: Self) -> Self {
        self * rhs.inv()
    }
}

impl Field {
    pub fn pow(self, exp: Self) -> Self {
        // repeated squaring
        let mut base = self;
        let mut exp = exp.0;
        let mut result: Field = Field(1);
        while exp > 0 {
            while (exp & 1) == 0 {
                exp /= 2;
                base = base * base;
            }
            exp -= 1;
            result = result * base;
        }
        result
    }

    pub fn inv(self) -> Self {
        // extended Euclidean
        let mut x1: i32 = 1;
        let mut a1: u32 = self.0;
        let mut x0: i32 = 0;
        let mut a2: u32 = MODULUS;
        let mut q: u32 = 0;

        while a2 != 0 {
            let x2 = x0 - (q as i32) * x1;
            x0 = x1;
            let a0 = a1;
            x1 = x2;
            a1 = a2;
            q = a0 / a1;
            a2 = a0 - q * a1;
        }
        if x1 < 0 {
            let (r, _) = MODULUS.overflowing_add(x1 as u32);
            Field(r)
        } else {
            Field(x1 as u32)
        }
    }
}

impl From<u32> for Field {
    fn from(x: u32) -> Self {
        Field(x % MODULUS)
    }
}

#[test]
fn test_arithmetic() {
    use rand::prelude::*;
    // add
    assert_eq!(Field(MODULUS - 1) + Field(1), 0.into());
    assert_eq!(Field(MODULUS - 2) + Field(2), 0.into());
    assert_eq!(Field(MODULUS - 2) + Field(3), 1.into());
    assert_eq!(Field(1) + Field(1), 2.into());
    assert_eq!(Field(2) + Field(MODULUS), 2.into());
    assert_eq!(Field(3) + Field(MODULUS - 1), 2.into());

    // sub
    assert_eq!(Field(0) - Field(1), (MODULUS - 1).into());
    assert_eq!(Field(1) - Field(2), (MODULUS - 1).into());
    assert_eq!(Field(15) - Field(3), 12.into());
    assert_eq!(Field(1) - Field(1), 0.into());
    assert_eq!(Field(2) - Field(MODULUS), 2.into());
    assert_eq!(Field(3) - Field(MODULUS - 1), 4.into());

    // add + sub
    for _ in 0..100 {
        let f = Field(random());
        let g = Field(random());
        assert_eq!(f + g - f - g, 0.into());
        assert_eq!(f + g - g, f);
        assert_eq!(f + g - f, g);
    }

    // mul
    assert_eq!(Field(35) * Field(123), 4305.into());
    assert_eq!(Field(1) * Field(MODULUS), 0.into());
    assert_eq!(Field(0) * Field(123), 0.into());
    assert_eq!(Field(123) * Field(0), 0.into());
    assert_eq!(Field(123123123) * Field(123123123), 1237630077.into());

    // div
    assert_eq!(Field(35) / Field(5), 7.into());
    assert_eq!(Field(35) / Field(0), 0.into());
    assert_eq!(Field(0) / Field(5), 0.into());
    assert_eq!(Field(1237630077) / Field(123123123), 123123123.into());

    assert_eq!(Field(0).inv(), 0.into());

    // mul and div
    let uniform = rand::distributions::Uniform::from(1..MODULUS);
    let mut rng = thread_rng();
    for _ in 0..100 {
        // non-zero element
        let f = Field(uniform.sample(&mut rng));
        assert_eq!(f * f.inv(), 1.into());
        assert_eq!(f.inv() * f, 1.into());
    }

    // pow
    assert_eq!(Field(2).pow(3.into()), 8.into());
    assert_eq!(Field(3).pow(9.into()), 19683.into());
    assert_eq!(Field(51).pow(27.into()), 3760729523.into());
}
