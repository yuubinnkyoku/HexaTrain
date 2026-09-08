#include <cassert>
#include <cmath>
#include <initializer_list>
#include <string>

#include "nicopedia_learning_rate_schedule.h"

int main() {
  using phonelm::nicopedia_schedule::Config;
  using phonelm::nicopedia_schedule::Kind;
  using phonelm::nicopedia_schedule::at;
  using phonelm::nicopedia_schedule::kindName;
  using phonelm::nicopedia_schedule::validate;

  Config s4000{Kind::LINEAR_DECAY, .0022f, .0015f, 4000, 8000, true};
  assert(validate(s4000, 8000));
  assert(at(s4000, 4000) == .0022f);
  assert(std::abs(at(s4000, 4001) - .002199825f) < 1.0e-9f);
  assert(std::abs(at(s4000, 6000) - .00185f) < 1.0e-9f);
  assert(std::abs(at(s4000, 7999) - .001500175f) < 1.0e-9f);
  assert(at(s4000, 8000) == .0015f);

  Config s6000{Kind::LINEAR_DECAY, .0022f, .0015f, 6000, 8000, true};
  assert(validate(s6000, 8000));
  assert(at(s6000, 6000) == .0022f);
  assert(std::abs(at(s6000, 6001) - .00219965f) < 1.0e-9f);
  assert(std::abs(at(s6000, 7000) - .00185f) < 1.0e-9f);
  assert(std::abs(at(s6000, 7999) - .00150035f) < 1.0e-9f);
  assert(at(s6000, 8000) == .0015f);

  // Schedule-v2a varies only the target endpoint; all candidates share the
  // same 6000->8000 linear shape and must hit their exact inclusive endpoint.
  for (const float target : {.0010f, .0007f, .0004f, .0002f, .0001f, .0000f}) {
    Config candidate{Kind::LINEAR_DECAY, .0022f, target, 6000, 8000, true};
    assert(validate(candidate, 8000));
    assert(at(candidate, 6000) == .0022f);
    assert(at(candidate, 8000) == target);
  }
  assert(std::abs(at(Config{Kind::LINEAR_DECAY, .0022f, .0010f, 6000, 8000, true}, 7000) - .0016f) < 1.0e-9f);
  assert(std::abs(at(Config{Kind::LINEAR_DECAY, .0022f, .0007f, 6000, 8000, true}, 7000) - .00145f) < 1.0e-9f);
  assert(std::abs(at(Config{Kind::LINEAR_DECAY, .0022f, .0004f, 6000, 8000, true}, 7000) - .0013f) < 1.0e-9f);
  Config zeroTarget{Kind::LINEAR_DECAY, .0022f, 0.0f, 6000, 8000, true};
  assert(validate(zeroTarget, 8000));
  assert(std::abs(at(zeroTarget, 7000) - .0011f) < 1.0e-9f);
  assert(at(zeroTarget, 8000) == 0.0f);

  // Schedule-v2c: only the cooldown shape changes.  The exact definition is
  // shape = 1 - sqrt(p), followed by non-zero-target affine scaling.
  Config sqrt6000{Kind::SQRT_DECAY, .0022f, .0001f, 6000, 8000, true};
  assert(validate(sqrt6000, 8000));
  assert(std::string(kindName(Kind::SQRT_DECAY)) == "sqrt_decay");
  assert(at(sqrt6000, 6000) == .0022f);
  assert(std::abs(at(sqrt6000, 6001) - .0021530425f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 6250) - .0014575379f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 6500) - .00115f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 6750) - .0009140179f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 7000) - .00071507576f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 7250) - .0005398042f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 7500) - .00038134665f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 7750) - .00023562988f) < 1.0e-9f);
  assert(std::abs(at(sqrt6000, 7999) - .00010052507f) < 1.0e-9f);
  assert(at(sqrt6000, 8000) == .0001f);

  // Non-zero-target scaling must be applied to the sqrt shape, rather than
  // treating the target as an additive post-processing offset.
  Config sqrtScaledTarget{Kind::SQRT_DECAY, .0022f, .0005f, 6000, 8000,
                          true};
  assert(validate(sqrtScaledTarget, 8000));
  assert(std::abs(at(sqrtScaledTarget, 7000) - .0009979184f) < 1.0e-9f);
  Config sqrtZeroTarget{Kind::SQRT_DECAY, .0022f, 0.0f, 6000, 8000, true};
  assert(validate(sqrtZeroTarget, 8000));
  assert(std::abs(at(sqrtZeroTarget, 7000) - .0006443651f) < 1.0e-9f);
  assert(at(sqrtZeroTarget, 8000) == 0.0f);
  assert(!validate(Config{Kind::SQRT_DECAY, .0022f, .0001f, 6000, 8000, false}, 8000));
  assert(!validate(Config{Kind::SQRT_DECAY, .0022f, .0001f, 6000, 6000, true}, 8000));
  assert(!validate(Config{Kind::SQRT_DECAY, .0022f, .0001f, 6000, 9000, true}, 8000));

  Config constant{Kind::CONSTANT, .0015f, .0015f, 0, 0, false};
  assert(validate(constant, 8000));
  assert(at(constant, 1) == .0015f);
  assert(at(constant, 8000) == .0015f);
  assert(!validate(Config{Kind::LINEAR_DECAY, .0022f, .0015f, 4000, 8000, false}, 8000));
  assert(!validate(Config{Kind::LINEAR_DECAY, .0022f, .0015f, 6000, 6000, true}, 8000));
  assert(!validate(Config{Kind::LINEAR_DECAY, .0022f, .0015f, 6000, 9000, true}, 8000));
  return 0;
}
