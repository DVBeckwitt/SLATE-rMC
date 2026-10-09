#include "cif_peaks.h"
#include <gemmi/cif.hpp>
#include <gemmi/smcif.hpp>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double hc_eV_A = 12398.419843320026;
struct Elastic {
    std::string species;
    uint32_t atomic_number;
    std::array<double, 11> coefficient;
};
struct Anomalous {
    uint32_t atomic_number;
    std::vector<std::array<double, 6>> knots;
};
struct Atom {
    gemmi::Fractional position;
    double occupancy, u_iso;
    size_t species;
};
struct Factor {
    Elastic elastic;
    std::complex<double> anomalous;
};
uint32_t crc32(const std::vector<unsigned char> &bytes, size_t start = 0) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = start; i < bytes.size(); ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
void require(bool condition, const std::string &error) {
    if (!condition)
        throw std::runtime_error(error);
}
void check(CifProgress progress, void *context) {
    if (progress && progress(context, "Calculating CIF Bragg positions and raw |F|^2"))
        throw std::runtime_error("Canceled.");
}
std::vector<unsigned char> read_bytes(const char *path, long limit) {
    std::unique_ptr<FILE, decltype(&fclose)> file(fopen(path, "rb"), fclose);
    require(file != nullptr, std::string("Cannot open: ") + path);
    require(fseek(file.get(), 0, SEEK_END) == 0, "Cannot seek input file.");
    long size = ftell(file.get());
    require(size > 0 && size <= limit, "Input is empty or exceeds its size limit.");
    require(fseek(file.get(), 0, SEEK_SET) == 0, "Cannot rewind input file.");
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    require(fread(bytes.data(), 1, bytes.size(), file.get()) == bytes.size(), "Input read failed.");
    return bytes;
}
template <typename T> T consume(const std::vector<unsigned char> &bytes, size_t &position) {
    require(position <= bytes.size() && sizeof(T) <= bytes.size() - position,
            "Truncated scattering data. Restore cif_scattering.bin from the distribution.");
    T value;
    memcpy(&value, bytes.data() + position, sizeof value);
    position += sizeof value;
    return value;
}
uint32_t read_factors(const char *path, std::vector<Elastic> &elastic,
                      std::vector<Anomalous> &anomalous) {
    auto bytes = read_bytes(path, 8 * 1024 * 1024);
    require(bytes.size() >= 20 && memcmp(bytes.data(), "SLCIF01\0", 8) == 0,
            "Unknown scattering data format. Restore cif_scattering.bin.");
    size_t position = 8;
    auto ne = consume<uint32_t>(bytes, position), na = consume<uint32_t>(bytes, position);
    auto crc = consume<uint32_t>(bytes, position);
    require(ne == 211 && na == 92 && crc32(bytes, position) == crc,
            "Scattering data failed its integrity check. Restore cif_scattering.bin.");
    for (uint32_t i = 0; i < ne; ++i) {
        auto name = consume<std::array<char, 16>>(bytes, position);
        require(name.back() == 0, "Invalid scattering species.");
        Elastic entry{std::string(name.data()), consume<uint32_t>(bytes, position),
                      consume<std::array<double, 11>>(bytes, position)};
        for (double value : entry.coefficient)
            require(std::isfinite(value), "Invalid elastic factor.");
        elastic.push_back(entry);
    }
    for (uint32_t i = 0; i < na; ++i) {
        Anomalous entry{consume<uint32_t>(bytes, position), {}};
        uint32_t n = consume<uint32_t>(bytes, position);
        require(entry.atomic_number == i + 1 && n >= 4 && n <= 3000,
                "Invalid anomalous-factor table.");
        for (uint32_t j = 0; j < n; ++j) {
            auto knot = consume<std::array<double, 6>>(bytes, position);
            for (double value : knot)
                require(std::isfinite(value), "Invalid anomalous factor.");
            require(knot[0] > 0 && (j == 0 || knot[0] > entry.knots.back()[0]),
                    "Invalid anomalous energy ordering.");
            entry.knots.push_back(knot);
        }
        anomalous.push_back(std::move(entry));
    }
    require(position == bytes.size(), "Unexpected scattering data content.");
    return crc;
}
std::complex<double> dispersion(const Anomalous &table, double energy) {
    const auto &knots = table.knots;
    require(energy >= 250 && energy <= knots.back()[0],
            "Wavelength outside the supported Chantler range (250 eV to table maximum).");
    auto upper =
        std::upper_bound(knots.begin(), knots.end(), energy,
                         [](double e, const std::array<double, 6> &k) { return e < k[0]; });
    size_t i = std::min(static_cast<size_t>(upper - knots.begin() - 1), knots.size() - 2);
    const auto &a = knots[i], &b = knots[i + 1];
    double t = (energy - a[0]) / (b[0] - a[0]);
    double f1 = a[1] + t * (a[2] + t * (a[3] + t * a[4]));
    double fraction = (std::log(energy) - std::log(a[0])) / (std::log(b[0]) - std::log(a[0]));
    return {f1, std::exp(a[5] + fraction * (b[5] - a[5]))};
}
void required_column(gemmi::cif::Block &block, const char *tag, size_t count) {
    auto column = block.find_values(tag);
    require(static_cast<size_t>(column.length()) == count,
            std::string("CIF must explicitly provide ") + tag);
    for (const auto &value : column)
        require(!gemmi::cif::is_null(value), std::string("Missing CIF value in ") + tag);
}
void validate_sites(gemmi::cif::Block &block, gemmi::SmallStructure &small, bool unknown_zero,
                    CifPeaks &result) {
    require(!small.sites.empty() && small.sites.size() <= 2000,
            "CIF requires 1 to 2000 source sites.");
    for (const char *tag : {"_atom_site_label", "_atom_site_type_symbol", "_atom_site_occupancy",
                            "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"})
        required_column(block, tag, small.sites.size());
    require(block.find_values("_atom_site_aniso_label").length() == 0,
            "Anisotropic displacement metadata is not supported.");
    for (const char *kind : {"U", "B"})
        for (const char *suffix : {"11", "22", "33", "12", "13", "23"})
            for (const auto &value :
                 block.find_values(std::string("_atom_site_aniso_") + kind + "_" + suffix))
                require(gemmi::cif::is_null(value),
                        "Anisotropic displacement metadata is not supported.");
    auto u = block.find_values("_atom_site_U_iso_or_equiv");
    auto b = block.find_values("_atom_site_B_iso_or_equiv");
    require(!(u && b), "Ambiguous CIF: both isotropic U and B columns are present.");
    auto displacement = u ? u : b;
    auto occupancies = block.find_values("_atom_site_occupancy");
    require(!displacement || static_cast<size_t>(displacement.length()) == small.sites.size(),
            "Displacement column does not align with atom sites.");
    std::set<std::string> labels;
    for (size_t i = 0; i < small.sites.size(); ++i) {
        auto &site = small.sites[i];
        site.occ = gemmi::cif::as_number(occupancies.at(i));
        require(!site.label.empty() && labels.insert(site.label).second,
                "CIF atom labels must be unique.");
        require(site.element.atomic_number() >= 1 && site.element.atomic_number() <= 92,
                "Full X-ray factors support elements H through U only.");
        std::string canonical = site.element_and_charge_symbol();
        std::string unit_charge = std::string(site.element.name()) + (site.charge > 0 ? "+" : "-");
        require(site.type_symbol == canonical ||
                    (std::abs(site.charge) == 1 && site.type_symbol == unit_charge),
                "Use an explicit element or ionic species such as Cu, Cu+ or Cu2+ in "
                "atom_site_type_symbol.");
        require(std::isfinite(site.occ) && site.occ >= 0 && site.occ <= 1,
                "Occupancy must be in [0,1].");
        for (double value : {site.fract.x, site.fract.y, site.fract.z})
            require(std::isfinite(value) && std::abs(value) <= 1000000,
                    "Invalid fractional atomic coordinate.");
        if (!displacement || gemmi::cif::is_null(displacement.at(i))) {
            require(unknown_zero,
                    "CIF has unknown Uiso. Explicitly enable 'Unknown Uiso = 0' to calculate.");
            site.u_iso = 0;
            ++result.unknown_u_sites;
        } else
            site.u_iso = gemmi::cif::as_number(displacement.at(i)) / (u ? 1 : 8 * pi * pi);
        require(std::isfinite(site.u_iso) && site.u_iso >= 0 && !site.aniso.nonzero(),
                "Displacements must be finite, nonnegative and isotropic.");
    }
}
gemmi::SmallStructure parse_structure(const std::vector<unsigned char> &source, bool unknown_zero,
                                      CifPeaks &result) {
    require(std::find(source.begin(), source.end(), 0) == source.end(), "CIF contains a NUL byte.");
    auto document = gemmi::cif::read_memory(reinterpret_cast<const char *>(source.data()),
                                            source.size(), "CIF", 2);
    require(document.blocks.size() == 1, "CIF must contain exactly one data block.");
    auto &block = document.sole_block();
    int site_count = block.find_values("_atom_site_label").length();
    require(site_count >= 1 && site_count <= 2000, "CIF requires 1 to 2000 source sites.");
    for (const char *tag : {"_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"})
        require(block.find_values(tag).length() <= 192, "CIF exceeds 192 symmetry operations.");
    const char *aliases[][2] = {{"_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"},
                                {"_space_group_name_Hall", "_symmetry_space_group_name_Hall"},
                                {"_space_group_IT_number", "_symmetry_Int_Tables_number"},
                                {"_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"}};
    for (const auto &pair : aliases) {
        auto first = block.find_values(pair[0]), second = block.find_values(pair[1]);
        if (first && second) {
            require(first.length() == second.length(), "Conflicting CIF symmetry tag aliases.");
            for (int i = 0; i < first.length(); ++i)
                require(gemmi::cif::as_string(first.at(i)) == gemmi::cif::as_string(second.at(i)),
                        "Ambiguous CIF symmetry aliases; supply one consistent declaration.");
        }
    }
    for (const char *tag : {"_cell_length_a", "_cell_length_b", "_cell_length_c",
                            "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"}) {
        required_column(block, tag, 1);
        double value = gemmi::cif::as_number(block.find_values(tag).at(0));
        require(std::isfinite(value) && value > 0, "Cell parameters must be finite and positive.");
    }
    auto small = gemmi::make_small_structure_from_block(block);
    small.determine_and_set_spacegroup("SH2N");
    require(small.spacegroup != nullptr, "CIF space group is missing or unresolved.");
    require(small.check_spacegroup().empty(),
            "Conflicting or incomplete CIF space-group declarations.");
    require(small.cell.alpha < 180 && small.cell.beta < 180 && small.cell.gamma < 180 &&
                std::isfinite(small.cell.volume) && small.cell.volume > 1e-9 &&
                small.cell.is_compatible_with_spacegroup(small.spacegroup),
            "CIF cell metric is invalid or incompatible with the space group.");
    validate_sites(block, small, unknown_zero, result);
    snprintf(result.name, sizeof result.name, "%s", small.name.c_str());
    snprintf(result.spacegroup, sizeof result.spacegroup, "%s", small.spacegroup->xhm().c_str());
    result.source_sites = static_cast<int>(small.sites.size());
    return small;
}
std::vector<Atom> expand_atoms(const gemmi::SmallStructure &small,
                               const std::vector<Elastic> &elastic,
                               const std::vector<Anomalous> &anomalous, double energy,
                               std::vector<Factor> &factors, CifPeaks &result, CifProgress progress,
                               void *context) {
    auto expanded = small.get_all_unit_cell_sites();
    require(expanded.size() <= 10000, "CIF exceeds 10000 expanded sites.");
    std::vector<Atom> atoms;
    std::string mappings;
    for (const auto &source : small.sites) {
        check(progress, context);
        std::vector<const gemmi::SmallStructure::Site *> orbit;
        for (const auto &site : expanded)
            if (site.label == source.label)
                orbit.push_back(&site);
        /* Qualify Gemmi's canonical 0.4 A orbit deduplication: no discarded
           image may differ by more than 0.0001 A from a retained image. */
        for (const auto &transform : small.cell.images) {
            auto position = transform.apply(source.fract);
            bool represented = false;
            for (const auto *site : orbit)
                if (small.cell.distance_sq(position, site->fract) <= 1e-8) {
                    represented = true;
                    break;
                }
            require(represented, "Ambiguous near-special atom position: refine coordinates or "
                                 "supply an explicit P1 cell.");
        }
    }
    for (const auto &site : expanded) {
        std::string wanted = site.element_and_charge_symbol();
        auto match = std::find_if(elastic.begin(), elastic.end(),
                                  [&](const Elastic &e) { return e.species == wanted; });
        if (match == elastic.end())
            match = std::find_if(elastic.begin(), elastic.end(), [&](const Elastic &e) {
                return e.species == site.element.name();
            });
        require(match != elastic.end(), "No elastic scattering factor for CIF species.");
        auto existing = std::find_if(factors.begin(), factors.end(), [&](const Factor &f) {
            return f.elastic.species == match->species;
        });
        size_t index = static_cast<size_t>(existing - factors.begin());
        if (existing == factors.end())
            factors.push_back({*match, dispersion(anomalous.at(match->atomic_number - 1), energy)});
        std::string mapping = site.type_symbol + "->" + match->species + ";";
        if (mappings.find(mapping) == std::string::npos)
            mappings += mapping;
        auto position = site.fract.wrap_to_unit();
        for (int axis = 0; axis < 3; ++axis)
            if (std::abs(position.at(axis) - 1) <= 1e-12)
                position.at(axis) = 0;
        atoms.push_back({position, site.occ, site.u_iso, index});
    }
    require(mappings.size() < sizeof result.species, "Too many scattering species mappings.");
    strcpy(result.species, mappings.c_str());
    result.expanded_sites = static_cast<int>(atoms.size());
    return atoms;
}
std::vector<CifPeak> reflections(const gemmi::UnitCell &cell, double wavelength, double maximum,
                                 size_t atom_count, CifProgress progress, void *context) {
    double reciprocal_max = 2 * std::sin(maximum * pi / 360) / wavelength;
    require(reciprocal_max <= 12,
            "Requested range exceeds Waasmaier validity: sin(theta)/lambda <= 6 / A.");
    std::array<int, 3> limit;
    double candidates = 1;
    int axis = 0;
    for (double length : {cell.a, cell.b, cell.c}) {
        double bound = std::ceil(length * reciprocal_max);
        require(bound <= 10000, "Reflection search too large. Reduce maximum 2theta.");
        limit[axis++] = static_cast<int>(bound);
        candidates *= 2 * bound + 1;
    }
    require(candidates <= 2000000, "More than 2000000 candidate hkl. Reduce maximum 2theta.");
    std::vector<CifPeak> peaks;
    for (int h = -limit[0]; h <= limit[0]; ++h) {
        check(progress, context);
        for (int k = -limit[1]; k <= limit[1]; ++k)
            for (int l = -limit[2]; l <= limit[2]; ++l) {
                if (h == 0 && k == 0 && l == 0)
                    continue;
                double reciprocal_sq = cell.calculate_1_d2({h, k, l});
                require(std::isfinite(reciprocal_sq) && reciprocal_sq > 0,
                        "Invalid reciprocal cell metric.");
                if (reciprocal_sq > reciprocal_max * reciprocal_max * (1 + 1e-14))
                    continue;
                double d = 1 / std::sqrt(reciprocal_sq);
                double angle = 2 * std::asin(std::min(1.0, wavelength / (2 * d))) * 180 / pi;
                peaks.push_back({h, k, l, d, angle, 0, 0, 0});
                require(peaks.size() <= 100000 && peaks.size() * atom_count <= 50000000,
                        "Calculation exceeds 100000 reflections or 50000000 atom terms. Reduce "
                        "maximum 2theta.");
            }
    }
    std::sort(peaks.begin(), peaks.end(), [](const CifPeak &a, const CifPeak &b) {
        if (a.two_theta_deg != b.two_theta_deg)
            return a.two_theta_deg < b.two_theta_deg;
        if (a.h != b.h)
            return a.h < b.h;
        if (a.k != b.k)
            return a.k < b.k;
        return a.l < b.l;
    });
    return peaks;
}
} // namespace

extern "C" int cif_calculate(const char *path, const char *data_path, double wavelength,
                             double maximum, int unknown_zero, CifPeaks *output,
                             CifProgress progress, void *context, char error[256]) {
    try {
        require(std::isfinite(wavelength) && wavelength > 0 && std::isfinite(maximum) &&
                    maximum > 0 && maximum < 180,
                "Use positive wavelength and maximum 2theta strictly between 0 and 180 degrees.");
        CifPeaks result{};
        result.wavelength_A = wavelength;
        result.max_two_theta_deg = maximum;
        auto source = read_bytes(path, 2 * 1024 * 1024);
        result.source_crc32 = crc32(source);
        check(progress, context);
        auto small = parse_structure(source, unknown_zero != 0, result);
        std::vector<Elastic> elastic;
        std::vector<Anomalous> anomalous;
        result.data_crc32 = read_factors(data_path, elastic, anomalous);
        std::vector<Factor> factors;
        auto atoms = expand_atoms(small, elastic, anomalous, hc_eV_A / wavelength, factors, result,
                                  progress, context);
        auto peaks = reflections(small.cell, wavelength, maximum, atoms.size(), progress, context);
        std::vector<std::complex<double>> values(factors.size());
        for (size_t i = 0; i < peaks.size(); ++i) {
            if (i % 64 == 0)
                check(progress, context);
            auto &peak = peaks[i];
            double s2 = 1 / (4 * peak.d_A * peak.d_A);
            for (size_t f = 0; f < factors.size(); ++f) {
                const auto &c = factors[f].elastic.coefficient;
                double f0 = c[0];
                for (int j = 0; j < 5; ++j)
                    f0 += c[1 + j] * std::exp(-c[6 + j] * s2);
                values[f] = f0 + factors[f].anomalous;
            }
            std::complex<double> sum = 0;
            for (const auto &atom : atoms) {
                double phase = 2 * pi *
                               (peak.h * atom.position.x + peak.k * atom.position.y +
                                peak.l * atom.position.z);
                double weight = atom.occupancy * std::exp(-8 * pi * pi * atom.u_iso * s2);
                sum += weight * values[atom.species] *
                       std::complex<double>(std::cos(phase), std::sin(phase));
            }
            peak.real_e = sum.real();
            peak.imag_e = sum.imag();
            peak.intensity_e2 = std::norm(sum);
            require(std::isfinite(peak.intensity_e2), "Nonfinite structure factor.");
        }
        check(progress, context);
        if (!peaks.empty()) {
            result.peaks = static_cast<CifPeak *>(malloc(peaks.size() * sizeof(CifPeak)));
            require(result.peaks != nullptr, "Not enough memory for CIF reflections.");
            memcpy(result.peaks, peaks.data(), peaks.size() * sizeof(CifPeak));
        }
        result.count = static_cast<int>(peaks.size());
        cif_peaks_free(output);
        *output = result;
        error[0] = 0;
        return 1;
    } catch (const std::exception &e) {
        snprintf(error, 256, "%s", e.what());
        return 0;
    } catch (...) {
        snprintf(error, 256, "CIF calculation failed.");
        return 0;
    }
}
extern "C" void cif_peaks_free(CifPeaks *result) {
    free(result->peaks);
    memset(result, 0, sizeof *result);
}
extern "C" int cif_peaks_write(FILE *stream, const CifPeaks *result, CifProgress progress,
                               void *context) {
    fprintf(
        stream,
        "# Raw per-signed-hkl F in electrons, |F|^2 in electrons^2; positive phase; 000 excluded\n"
        "# No multiplicity, normalization, Lorentz/polarization, absorption or detector "
        "correction\n"
        "# Waasmaier + Chantler; XrayDB 4.5.8 database 9.2; Gemmi 0.7.5\n"
        "# wavelength_A=%.17g,max_two_theta_deg=%.17g,source_crc32=%08lx,data_crc32=%08lx\n"
        "# source_sites=%d,expanded_sites=%d,unknown_Uiso_set_to_zero=%d\n"
        "# elastic_species=%s; anomalous factors use element; CIF dispersion values are not used\n"
        "h,k,l,d_A,two_theta_deg,F_real_e,F_imag_e,raw_SF_intensity_e2\n",
        result->wavelength_A, result->max_two_theta_deg, result->source_crc32, result->data_crc32,
        result->source_sites, result->expanded_sites, result->unknown_u_sites, result->species);
    for (int i = 0; i < result->count; ++i) {
        if (i % 256 == 0 && progress && progress(context, "Exporting CIF peaks"))
            return 0;
        const auto &p = result->peaks[i];
        fprintf(stream, "%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n", p.h, p.k, p.l, p.d_A,
                p.two_theta_deg, p.real_e, p.imag_e, p.intensity_e2);
    }
    return !ferror(stream);
}
