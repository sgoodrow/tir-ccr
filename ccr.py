#!/usr/bin/env python
import sys
from ccr_tools		import *

"""Code embodies the mathematical developments presented in two companion
papers submitted to Applied Optics in August 2012.  References to math in
the first paper on polarization and diffraction states are denoted as
pertaining to P1 (paper 1; Murphy & Goodrow), while the math related to
thermal gradients refers to P2 (paper 2; Goodrow & Murphy).  """

#========================== DEFAULTS ===========================#
b		= 0.						# initial minor-to-major axis ratio
f		= radians(0.0)					# angle between major axis and horizontal (radians)
d		= SENSE['LH']					# polarization rotation sense (LH or RH when viewed along ray)
incidence	= radians(0.0)					# angle of incidence to CCR face (radians)
azimuth		= radians(0.0)					# observer position around CCR (radians) (0 at x-axis)
dTr		= 0.						# radial thermal gradient (K)
dTz		= 0.						# axial thermal gradient (K)
#===============================================================#

#============================ INPUT ============================#
argument = sys.argv
num_args = len(argument)
if num_args == 1:
  print "Usage: %s b/a psi direc i az dTr dTz" % sys.argv[0]
  print "	Param	Default	Description"
  print "	b/a	0.0	minor-to-major axis ratio"
  print "	psi	0.0	orientation of major axis relative to horizontal axis in units of degrees"
  print "	direc	LH	LH or RH rotational sense, irrelevant for b/a = 0"
  print "	i	0.0	the angle of incidence in units of degrees, beware angles beyond 17 degrees"
  print "	az	0.0	azimuth of observer from x-axis in units of degrees"
  print "	dTr	0.0	thermal gradiant from center to rim in units of Kelvin"
  print "	dTz	0.0	thermal gradient from face to vertex in units of Kelvin"
  print
  print "Partial arguments okay; missing trailing arguments take defaults"
  sys.exit()
if num_args > 1: b		= float(argument[1])
if num_args > 2: f		= radians(float(argument[2]))
if num_args > 3: d		= SENSE[argument[3]]
if num_args > 4: incidence	= radians(float(argument[4]))
if num_args > 5: azimuth	= radians(float(argument[5]))
if num_args > 6: dTr		= float(argument[6]) / RADIUS
if num_args > 7: dTz		= float(argument[7]) / HEIGHT

print "Polarization:"
print "	% 0.2f	minor-to-major-axis ratio" 	% b
print "	% 0.2f	angle of major axis" 		% degrees(f)
print "	 %s  	rotation sense" 		% SENSEREV[d]
print
print "Cornercube:"
print "	% 0.2f	angle of incidence" 		% degrees(incidence)
print "	% 0.2f	observer position" 		% degrees(azimuth)
print
print "Thermal Difference:"
print "	% 0.2f	radial difference"	% (dTr * RADIUS)
print "	% 0.2f	axial difference" 	% (dTz * HEIGHT)
#===============================================================#

#======================= SETUP GEOMETRY ========================#
k = rotate([0, 0, -1], azimuth, incidence)			# initial ray direction
A = Surface([-1, -np.sqrt(3), np.sqrt(2)], 0)
B = Surface([ 2,           0, np.sqrt(2)], 0)
C = Surface([-1,  np.sqrt(3), np.sqrt(2)], 0)
F = Surface([ 0,           0,          1], HEIGHT)
paths = [[A, C, B], [A, B, C], [B, A, C], [B, C, A], [C, B, A], [C, A, B]]
surfaces = [A, B, C]
#===============================================================#

#========================= POLARIZATON =========================#
# observer frame
hor_axis_i = rotate([0, 1, 0], azimuth)				# Equation (2)
ver_axis_i = np.cross(hor_axis_i, k)

# plot initialization
ax = setup_plot(b, f, d, azimuth)

# initialize
pathrays = list()

# calculate path properties
for num, path in enumerate(paths):
	ray = Ray(k, b, f, d, path)
	ray.refract(F, N_ENV, N_CCR)

	hor_axis = ray.reflect(path[0], N_CCR, N_ENV, hor_axis_i, TIR)
	hor_axis = ray.reflect(path[1], N_CCR, N_ENV, hor_axis, TIR)
	hor_axis = ray.reflect(path[2], N_CCR, N_ENV, hor_axis, TIR)

	ray.refract(F, N_CCR, N_ENV)

	# rotate to observer frame
	ver_axis = np.cross(hor_axis, ray.dir)
	t = math.atan2(np.dot(-ver_axis_i, hor_axis), np.dot(ver_axis_i, ver_axis))
	#t -= radians(0) 	# like rotating a linear analyzer, we may wish to rotate 
				# our observer frame to enable complete component separation.

	ray.rotate_polarization(t)
	ray.phs_s -= PI						# flip handedness

	# save for diffraction patterns
	pathrays.append(ray)

	# get ellipse
	ray.get_ellipse()

	# plot ellipse
	add_to_plot(ax, num, azimuth, ray)

# ensure images directory exists
directory_path = os.path.join(os.getcwd(), 'images/')
directory = os.path.dirname(directory_path)
if not os.path.exists(directory):
	os.makedirs(directory)
	
# save ellipses
pl.savefig(os.path.join(directory_path, 'ellipses.png'))
#===============================================================#

#===================== DIFFRACTION PATTERN =====================#
# set up apertures and wavefronts
ap_msk  = np.zeros([SIZE, SIZE], dtype='d') + 1
ap_hor  = np.zeros([SIZE, SIZE], dtype='d')
ap_ver  = np.zeros([SIZE, SIZE], dtype='d')
wf_hor  = np.zeros([SIZE, SIZE], dtype='d')
wf_ver  = np.zeros([SIZE, SIZE], dtype='d')
thermal = np.zeros([SIZE, SIZE], dtype='d')

# prepare basis vectors for wavefront
u, v = get_basis(k)

# begin ray trace
for i in range(SIZE):
	for j in range(SIZE):
		ray = Ray(k, b, f, d, None, (i, j, u, v))
		if not ray.hits_front(F, dTr, dTz):
			continue
		ray.num_waves = 0
		ray.refract(F, N_ENV, N_CCR)
		ray.reflect(ray.collide(surfaces, dTr, dTz), N_CCR, N_ENV)
		ray.reflect(ray.collide(surfaces, dTr, dTz), N_CCR, N_ENV)
		ray.reflect(ray.collide(surfaces, dTr, dTz), N_CCR, N_ENV)
		if not ray.hits_front(F, dTr, dTz):
			continue
		ray.refract(F, N_CCR, N_ENV)
		index = paths.index(ray.surfs_hit)
		ap_hor[-i, j] = round(pathrays[index].amp_s, 10)
		ap_ver[-i, j] = round(pathrays[index].amp_p, 10)
		wf_hor[-i, j] = round(pathrays[index].phs_s, 10)
		wf_ver[-i, j] = round(pathrays[index].phs_p, 10)
		thermal[-i, j] = ray.num_waves * 2 * PI

# set thermal wavefront minimum to 0
locs = np.nonzero(thermal)[0]
if len(locs): thermal[locs] -= thermal[locs].min()

# add thermal wavefront to wavefront
wf_hor -= thermal
wf_ver -= thermal

# get diffraction patterns
df_hor = diffract(ap_hor, wf_hor)
df_ver = diffract(ap_ver, wf_ver)
df_tot = df_hor + df_ver

# get flux in first airy ring
outf = ((ap_hor**2).sum() + (ap_ver**2).sum()) * (SIZE * LOD)**2
airy = flux(df_tot, [SIZE / 2, SIZE / 2], 1.22 * LOD) / outf * 100
print
print "Results:"
print "	% 0.2f	airy ring flux"	% (airy)
print "	% 0.2f	central irradiance" 	% (df_tot.max() / outf * 100)
#===============================================================#

#======================= Draw Images ===========================#
make_image(thermal, 'thermal', 'fp', directory)
make_image(ap_hor, 'ap_hor', 'fp', directory)
make_image(ap_ver, 'ap_ver', 'fp', directory)
make_image(wf_hor, 'wf_hor', 'fp', directory)
make_image(wf_ver, 'wf_ver', 'fp', directory)
make_image(df_hor, 'df_hor', 'fp', directory, invert = True, linearize = False)
make_image(df_ver, 'df_ver', 'fp', directory, invert = True, linearize = False)
make_image(df_tot, 'df_tot', 'fp', directory, invert = True, linearize = False)
#===============================================================#
