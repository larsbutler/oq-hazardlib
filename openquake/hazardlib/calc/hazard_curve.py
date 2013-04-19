# The Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
:mod:`openquake.hazardlib.calc.hazard_curve` implements
:func:`hazard_curves_poissonian`.
"""
import numpy

from itertools import izip

from openquake.hazardlib.calc import filters
from openquake.hazardlib.geo import Point
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.source import AreaSource
from openquake.hazardlib.source import PointSource
from openquake.hazardlib.tom import PoissonTOM


def hazard_curves_poissonian(
        sources, sites, imts, time_span, gsims, truncation_level,
        source_site_filter=filters.source_site_noop_filter,
        rupture_site_filter=filters.rupture_site_noop_filter):
    """
    Compute hazard curves on a list of sites, given a set of seismic sources
    and a set of ground shaking intensity models (one per tectonic region type
    considered in the seismic sources).

    The calculator assumes
    :class:`Poissonian <openquake.hazardlib.tom.PoissonTOM>` temporal
    occurrence model.

    The calculator computes probability of ground motion exceedance according
    to the equation as described in pag. 419 of "OpenSHA: A Developing
    Community-modeling Environment for Seismic Hazard Analysis, Edward
    H. Field, Thomas H. Jordan and C. Allin Cornell. Seismological Research
    Letters July/August 2003 v. 74 no. 4 p. 406-419".

    :param sources:
        An iterator of seismic sources objects (instances of subclasses
        of :class:`~openquake.hazardlib.source.base.SeismicSource`).
    :param sites:
        Instance of :class:`~openquake.hazardlib.site.SiteCollection` object,
        representing sites of interest.
    :param imts:
        Dictionary mapping intensity measure type objects (see
        :mod:`openquake.hazardlib.imt`) to lists of intensity measure levels.
    :param time_span:
        An investigation period for Poissonian temporal occurrence model,
        floating point number in years.
    :param gsims:
        Dictionary mapping tectonic region types (members
        of :class:`openquake.hazardlib.const.TRT`) to
        :class:`~openquake.hazardlib.gsim.base.GMPE` or
        :class:`~openquake.hazardlib.gsim.base.IPE` objects.
    :param trunctation_level:
        Float, number of standard deviations for truncation of the intensity
        distribution.
    :param source_site_filter:
        Optional source-site filter function. See
        :mod:`openquake.hazardlib.calc.filters`.
    :param rupture_site_filter:
        Optional rupture-site filter function. See
        :mod:`openquake.hazardlib.calc.filters`.

    :returns:
        Dictionary mapping intensity measure type objects (same keys
        as in parameter ``imts``) to 2d numpy arrays of float, where
        first dimension differentiates sites (the order and length
        are the same as in ``sites`` parameter) and the second one
        differentiates IMLs (the order and length are the same as
        corresponding value in ``imts`` dict).
    """
    curves = dict((imt, numpy.ones([len(sites), len(imts[imt])]))
                  for imt in imts)
    tom = PoissonTOM(time_span)

    total_sites = len(sites)

    sources = prepare_sources(sources)
    sources_sites = ((source, sites) for source in sources)

    for source, s_sites in source_site_filter(sources_sites):
        ruptures_sites = ((rupture, s_sites)
                          for rupture in source.iter_ruptures(tom))
        for rupture, r_sites in rupture_site_filter(ruptures_sites):
            prob = rupture.get_probability_one_or_more_occurrences()
            gsim = gsims[rupture.tectonic_region_type]
            sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
            for imt in imts:
                poes = gsim.get_poes(sctx, rctx, dctx, imt, imts[imt],
                                     truncation_level)
                curves[imt] *= r_sites.expand(
                    (1 - prob) ** poes, total_sites, placeholder=1
                )

    for imt in imts:
        curves[imt] = 1 - curves[imt]
    return curves


def prepare_sources(sources):
    for source in sources:
        if isinstance(source, AreaSource):
            for pt_src in area_source_to_point_sources(source):
                yield pt_src
        else:
            yield source

def area_source_to_point_sources(area_source):
    mesh = area_source.polygon.discretize(area_source.area_discretization)
    num_points = len(mesh)

    mfd = area_source.mfd
    if isinstance(mfd, TruncatedGRMFD):
        new_a_val = numpy.log10(10 ** area_source.mfd.a_val / num_points)
        new_mfd = TruncatedGRMFD(min_mag=mfd.min_mag, max_mag=mfd.max_mag,
                                 bin_width=mfd.bin_width, a_val=new_a_val,
                                 b_val=mfd.b_val)
    elif isinstance(mfd, EvenlyDiscretizedMFD):
        new_occur_rates = [float(x) / num_points for x in mfd.occurrence_rates]
        new_mfd = EvenlyDiscretizedMFD(min_mag=mfd.min_mag,
                                       bin_width=mfd.bin_width,
                                       occurrence_rates=new_occur_rates)

    for lon, lat in izip(mesh.lons, mesh.lats):
        pt = PointSource(
            source_id='fake',
            name='fake',
            tectonic_region_type=area_source.tectonic_region_type,
            mfd=new_mfd,
            rupture_mesh_spacing=area_source.rupture_mesh_spacing,
            magnitude_scaling_relationship=area_source\
                .magnitude_scaling_relationship,
            rupture_aspect_ratio=area_source.rupture_aspect_ratio,
            upper_seismogenic_depth=area_source.upper_seismogenic_depth,
            lower_seismogenic_depth=area_source.lower_seismogenic_depth,
            location=Point(lon, lat),
            nodal_plane_distribution=area_source.nodal_plane_distribution,
            hypocenter_distribution=area_source.hypocenter_distribution
        )
        yield pt
