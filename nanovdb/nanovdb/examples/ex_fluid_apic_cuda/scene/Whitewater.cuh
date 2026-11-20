// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef FLUID_SCENE_WHITEWATER_CUH
#define FLUID_SCENE_WHITEWATER_CUH
#include "Character.cuh"
namespace fluid
{
struct Secondary
{
    Vec x{0}, v{0};
    Real radius = .004, age = 0, life = 1;
    uint64_t id = 0;
    int kind = 1;
};
__device__ double densityCell(Domain d, Coord c, double ppc)
{
    int i = d.index(c);
    if (!i)
        return 0;
    double mass = 0;
    for (int a = 0; a < 3; ++a)
    {
        mass += d.cells[i].mass[a];
        int j = d.neighbor(i, 2 * a + 1);
        if (j)
            mass += d.cells[j].mass[a];
    }
    return mass / (6 * ppc);
}
__device__ double density(Domain d, Vec p, double ppc)
{
    Vec q = p / d.h - Vec(.5);
    Coord c = floorCoord(q);
    Vec f = q - Vec(c[0], c[1], c[2]);
    double rho = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                rho += (x ? f[0] : 1 - f[0]) * (y ? f[1] : 1 - f[1]) * (z ? f[2] : 1 - f[2]) *
                       densityCell(d, c + Coord(x, y, z), ppc);
    return rho;
}
__device__ Vec densityGradient(Domain d, Vec p, double ppc)
{
    Vec g;
    for (int a = 0; a < 3; ++a)
    {
        Vec e(0);
        e[a] = .5 * d.h;
        g[a] = (density(d, p + e, ppc) - density(d, p - e, ppc)) / d.h;
    }
    return g;
}
// Do not mistake the liquid/solid density drop for a liquid/air surface.
__device__ double surfaceDensity(Domain d, MeshView mesh, Vec p, double ppc, bool closedSides)
{
    double rho = density(d, p, ppc);
    bool side = closedSides &&
                (p[0] < .5 * d.h || p[0] > (d.nx - .5) * d.h || p[2] < .5 * d.h || p[2] > (d.nz - .5) * d.h);
    return side || p[1] < .5 * d.h || mesh.field.sdf(p) < 0 ? maxValue(1., rho) : rho;
}
__device__ void floatOnSurface(Domain d, MeshView mesh, Vec &p, double ppc, bool closedSides = false)
{
    for (int j = 0; j < 8; ++j)
    {
        Vec g;
        for (int a = 0; a < 3; ++a)
        {
            Vec e(0);
            e[a] = .5 * d.h;
            g[a] = (surfaceDensity(d, mesh, p + e, ppc, closedSides) -
                    surfaceDensity(d, mesh, p - e, ppc, closedSides)) /
                   d.h;
        }
        double len = g.length();
        if (len < 1e-6)
            break;
        double delta = (surfaceDensity(d, mesh, p, ppc, closedSides) - .5) / len;
        p -= g / len * maxValue(-.3 * d.h, minValue(.3 * d.h, delta));
    }
}
__device__ bool sampleVelocity(Domain d, Vec p, Vec &v)
{
    bool found = true;
    for (int a = 0; a < 3; ++a)
    {
        Vec offset(.5);
        offset[a] = 0;
        Coord base = floorCoord(p / d.h - offset - Vec(.5));
        double sum = 0, value = 0;
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y)
                for (int z = 0; z < 3; ++z)
                {
                    Coord c = base + Coord(x, y, z);
                    int i = d.index(c);
                    if (!i || !d.cells[i].valid[a])
                        continue;
                    Vec r = face(c, a) - p / d.h;
                    double w = weight(r[0]) * weight(r[1]) * weight(r[2]);
                    sum += w;
                    value += w * d.cells[i].transport[a];
                }
        v[a] = sum > 1e-8 ? value / sum : 0;
        found = found && sum > .5;
    }
    return found;
}
__hostdev__ uint64_t mix(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
__hostdev__ double random01(uint64_t x)
{
    return (mix(x) >> 11) * 0x1.0p-53;
}
struct WWCheck
{
    int sdf = 0, mesh = 0, floor = 0, outside = 0, contacts = 0;
    double depth = 0, normalError = 0;
};
__global__ void spawnWhitewater(Domain d, const Particle *p, const Vec *before, int n, double dt,
                                double ppc, double volume, double rate, uint64_t step, Secondary *births,
                                int capacity, int *count, double *expected, double foamRadiusScale = 1,
                                double foamLifeScale = 1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    auto a = p[i];
    double rho = density(d, a.x, ppc);
    Vec grad = densityGradient(d, a.x, ppc);
    double gl = grad.length();
    // A physical 6 cm surface layer, with acceleration relative to ballistic
    // motion and vorticity as sources. Absolute speed is deliberately not a
    // source.
    if (rho < .08 || rho > 1.5 || gl < 1e-8 || fabs(rho - .5) / gl > .06)
        return;
    Vec dv = (a.v - before[i]) / dt - Vec(0, -9.81, 0);
    double vort = Vec(a.c[2][1] - a.c[1][2], a.c[0][2] - a.c[2][0], a.c[1][0] - a.c[0][1]).length();
    double drive = minValue(3., maxValue(0., (dv.length() - 20) / 45) + maxValue(0., (vort - 12) / 30));
    double lambda = rate * volume * dt * drive;
    atomicAdd(expected, lambda);
    uint64_t seed = mix(uint64_t(i) + step * 0x9e3779b97f4a7c15ULL);
    int number = int(lambda) + (random01(seed) < lambda - int(lambda));
    for (int k = 0; k < number; ++k)
    {
        int slot = atomicAdd(count, 1);
        if (slot >= capacity)
            continue;
        Secondary q;
        q.id = mix(seed + k);
        Vec normal = -grad / gl;
        q.kind = random01(seed + 2 + k) < .6 ? 1 : 0;
        q.radius = q.kind ? .0025 + .0025 * random01(seed + 3 + k) : .004 + .003 * random01(seed + 3 + k);
        q.life = q.kind ? .45 + .65 * random01(seed + 4 + k) : 1.1 + 1.7 * random01(seed + 4 + k);
        if (q.kind == 0)
        {
            q.radius *= foamRadiusScale;
            q.life *= foamLifeScale;
        }
        q.x = a.x + normal * .005;
        q.v = a.v + (q.kind ? normal * (.25 + .4 * random01(seed + 5 + k)) : Vec(0));
        births[slot] = q;
    }
}
__global__ void updateWhitewater(Domain d, MeshView mesh, Secondary *p, int n, double dt, double ppc,
                                 WWCheck *checks, bool independent, bool closedSides = false,
                                 double foamRadiusScale = 1, double foamLifeScale = 1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    auto &q = p[i];
    WWCheck check;
    Vec u;
    bool available = sampleVelocity(d, q.x, u);
    if (dt == 0)
        floatOnSurface(d, mesh, q.x, ppc, closedSides);
    if (dt > 0)
    {
        q.age += dt;
        if (q.kind == 1)
        {
            q.v[1] -= 9.81 * dt;
            q.v *= exp(-.35 * dt);
            q.x += q.v * dt;
            // Allow newborn/outgoing droplets to escape before testing re-entry.
            Vec grad = densityGradient(d, q.x, ppc);
            if (q.age > .08 && density(d, q.x, ppc) > .7 && available && (q.v - u).dot(grad) > 0)
            {
                q.kind = 0;
                q.radius = maxValue(.004, q.radius) * foamRadiusScale;
                q.life = q.age + (1.1 + 1.7 * random01(q.id)) * foamLifeScale;
            }
        }
        else if (available)
        {
            q.v = u;
            q.x += u * dt;
            floatOnSurface(d, mesh, q.x, ppc, closedSides);
        }
        else
        {
            q.kind = 1;
            q.v[1] -= 9.81 * dt;
            q.x += q.v * dt;
        }
    }
    double margin = q.radius + .002, phi = mesh.field.sdf(q.x);
    bool contact = false;
    for (int j = 0; j < 16 && phi < margin; ++j)
    {
        Vec normal = mesh.field.normal(q.x);
        if (normal.length() < .5)
        {
            auto near = mesh.nearest(q.x);
            normal = near.normal;
            q.x = near.point + normal * margin;
        }
        else
            q.x += normal * minValue(margin - phi, mesh.field.band);
        phi = mesh.field.sdf(q.x);
        contact = true;
    }
    // A concave, sampled SDF can still be negative outside the triangle union
    // after iterative normal pushes. Resolve both predicates, not mesh parity
    // alone.
    contact = mesh.resolveInterior(q.x, margin, true) || contact;
    const bool floorMoved=mesh.resolveFloor(q.x,margin,q.radius);
    if(floorMoved && !contact && q.x[1]<=q.radius+contactTolerance)
        q.v[1]=maxValue(Real(0),q.v[1]);
    phi = mesh.field.sdf(q.x);
    if (contact)
    {
        Vec normal = mesh.field.normal(q.x);
        auto near = mesh.nearest(q.x);
        if (normal.length() < .5)
            normal = near.normal;
        double rel = (q.v - near.velocity).dot(normal);
        q.v = contactVelocity(q.v, normal, near.velocity, q.x[1] <= q.radius + 1e-10);
        check.normalError = maxValue(0., -(q.v - near.velocity).dot(normal));
        check.contacts = 1;
    }
    if (q.x[1] < q.radius)
    {
        q.x[1] = q.radius;
        q.v[1] = maxValue(0., q.v[1]);
        q.age += 2 * dt;
    }
    check.sdf = phi < -contactTolerance;
    if (closedSides)
    {
        // Same stationary, frictionless domain walls as the APIC particles.
        for (int a = 0; a < 3; a += 2)
        {
            double hi = (a == 0 ? d.nx : d.nz) * d.h - q.radius;
            if (q.x[a] < q.radius)
            {
                q.x[a] = q.radius;
                q.v[a] = maxValue(0., q.v[a]);
            }
            if (q.x[a] > hi)
            {
                q.x[a] = hi;
                q.v[a] = minValue(0., q.v[a]);
            }
        }
    }
    check.floor = q.x[1] < 0;
    check.depth = maxValue(0., -phi);
    check.outside = !available;
    if (independent && mesh.inside(q.x))
    {
        auto near = mesh.nearest(q.x);
        if (near.distance > 1e-6)
        {
            check.mesh = 1;
            check.depth = maxValue(check.depth, near.distance);
        }
    }
    checks[i] = check;
    if (check.mesh || check.sdf || check.floor)
        printf("WW CONTACT FAIL id=%llu kind=%d x=(%.12g,%.12g,%.12g) radius=%g "
               "phi=%g mesh=%d sdf=%d age=%g\n",
               (unsigned long long)q.id, q.kind, q.x[0], q.x[1], q.x[2], q.radius, phi, check.mesh, check.sdf,
               q.age);
}
struct WWCombine
{
    __hostdev__ WWCheck operator()(WWCheck a,WWCheck b) const {
        a.sdf+=b.sdf;a.mesh+=b.mesh;a.floor+=b.floor;a.outside+=b.outside;a.contacts+=b.contacts;
        a.depth=maxValue(a.depth,b.depth);a.normalError=maxValue(a.normalError,b.normalError);return a;
    }
};
struct KeepSecondary
{
    __device__ bool operator()(const Secondary &q) const {
        return q.age<q.life && q.x[0]>=-.5 && q.x[0]<=3.7 && q.x[2]>=-.5 && q.x[2]<=2.9;
    }
};
struct WWMeasureResult {int foam=0,spray=0,invalid=0;double speed=0;};
struct WWMeasure
{
    const Secondary *p;
    __device__ WWMeasureResult operator()(int i) const {
        auto q=p[i];return {q.kind==0,q.kind!=0,!isfinite(q.x[0]+q.x[1]+q.x[2]+q.v.length()),double(q.v.length())};
    }
};
struct WWMeasureAdd
{
    __hostdev__ WWMeasureResult operator()(WWMeasureResult a,WWMeasureResult b) const {
        return {a.foam+b.foam,a.spray+b.spray,a.invalid+b.invalid,maxValue(a.speed,b.speed)};
    }
};
struct Whitewater
{
    bool closedSides=false,initialized=false;
    double foamRadiusScale=1,foamLifeScale=1;
    std::vector<Secondary> host; // explicit output/restart staging only
    Buffer<Secondary> data,scratch,births;
    Buffer<WWCheck> checks;
    int live=0;
    uint64_t spawned=0,expired=0;
    double expected=0;
    int foam=0,spray=0,outside=0,remaining=0;
    double maxSpeed=0,maxDepth=0;
    void initialize() {
        if(initialized) return;
        live=int(host.size());growBuffer(data,live);
        if(live) cudaCheck(cudaMemcpy(data.data(),host.data(),live*sizeof(Secondary),cudaMemcpyHostToDevice));
        initialized=true;
    }
    void download() {
        initialize();host.resize(live);
        if(live) cudaCheck(cudaMemcpy(host.data(),data.data(),live*sizeof(Secondary),cudaMemcpyDeviceToHost));
    }
    double step(Solver &s,MeshView mesh,const Buffer<Vec>& before,double dt,double ppc,double rate,uint64_t step)
    {
        Timer timer;initialize();growBuffer(births,4096);
        auto count=allocate<int>(1);auto expectation=allocate<double>(1);
        int added=0;double expect=0;
        for(;;) {
            cudaCheck(cudaMemset(count.data(),0,sizeof(int)));cudaCheck(cudaMemset(expectation.data(),0,sizeof(double)));
            spawnWhitewater<<<blocks(s.particleCount),256>>>(s.d,s.particles.data(),before.data(),s.particleCount,
                dt,ppc,pow(s.d.h,3)/ppc,rate,step,births.data(),births.size(),count.data(),expectation.data(),foamRadiusScale,foamLifeScale);
            cudaCheck(cudaMemcpy(&added,count.data(),sizeof(int),cudaMemcpyDeviceToHost));
            if(added<=int(births.size())) break;
            // Retry the deterministic source, never silently drop overflow births.
            growBuffer(births,added);
        }
        cudaCheck(cudaMemcpy(&expect,expectation.data(),sizeof(double),cudaMemcpyDeviceToHost));expected+=expect;
        remaining=outside=0;
        auto update=[&](Secondary *p,int n,double stepDt) {
            if(!n) return;
            growBuffer(checks,n);
            updateWhitewater<<<blocks(n),256>>>(s.d,mesh,p,n,stepDt,ppc,checks.data(),true,closedSides,foamRadiusScale,foamLifeScale);
            auto result=thrust::reduce(thrust::device,checks.data(),checks.data()+n,WWCheck{},WWCombine{});
            remaining+=result.sdf+result.mesh+result.floor;outside+=result.outside;maxDepth=std::max(maxDepth,result.depth);
        };
        update(data.data(),live,dt);update(births.data(),added,0);
        growBuffer(data,live+added,live);
        if(added) cudaCheck(cudaMemcpyAsync(data.data()+live,births.data(),added*sizeof(Secondary),cudaMemcpyDeviceToDevice));
        live+=added;spawned+=added;growBuffer(scratch,live);
        if(live) {
            auto end=thrust::copy_if(thrust::device,data.data(),data.data()+live,scratch.data(),KeepSecondary{});
            int retained=int(end-scratch.data());expired+=live-retained;live=retained;std::swap(data,scratch);
        }
        auto stats=reduceDevice(live,WWMeasure{data.data()},WWMeasureResult{},WWMeasureAdd{});
        foam=stats.foam;spray=stats.spray;maxSpeed=stats.speed;
        if(stats.invalid || remaining) throw std::runtime_error("invalid/penetrating whitewater");
        return timer.ms();
    }
};
} // namespace fluid
#endif
