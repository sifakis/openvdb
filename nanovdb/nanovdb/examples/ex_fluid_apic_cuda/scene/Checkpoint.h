// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Local binary restart format. It preserves APIC affine matrices and secondary
// state; visualization CSVs deliberately are not accepted as restart files.
namespace fluid
{
struct CheckpointHeader
{
    uint64_t magic = 0x464c554944303034ULL, particleSize = sizeof(Particle),
             secondarySize = sizeof(Secondary);
    uint64_t count = 0, secondaryCount = 0, nextId = 0, emitted = 0, drained = 0, spawned = 0, expired = 0;
    double time = 0, fraction = 0, expected = 0, h = 0, ppc = 0, radius = 0, speed = 0, emitterX = 0,
           emitterZ = 0;
    int nextFrame = 0, step = 0;
    uint64_t drainEnabled = 1;
    double initialDepth = 0, foamRadiusScale = 1, foamLifeScale = 1, whitewaterRate = 40000;
    uint64_t initialCount = 0, initialPoolCount = 0;
};
static_assert(offsetof(CheckpointHeader, nextFrame) == 152,
              "update streaming mesher checkpoint prefix reader");
inline void trimCsv(const std::string &path, double time)
{
    std::ifstream in(path);
    std::string line, all;
    std::getline(in, line);
    all = line + '\n';
    while (std::getline(in, line))
    {
        auto comma = line.find(',');
        auto second = line.find(',', comma + 1);
        double t = std::stod(line.substr(comma + 1, second - comma - 1));
        if (t <= time + 1e-10)
            all += line + '\n';
    }
    in.close();
    std::ofstream out(path);
    out << all;
}
} // namespace fluid
