
"""Code embodies the mathematical developments presented in two companion
papers submitted to Applied Optics in August 2012.  References to math in
the first paper on polarization and diffraction states are denoted as
pertaining to P1 (paper 1; Murphy & Goodrow), while the math related to
thermal gradients refers to P2 (paper 2; Goodrow & Murphy).  """

import os
import sys
import math
import cmath
import copy	as cp
import numpy 	as np
import matplotlib as mpl
mpl.use('Agg')
import pylab 	as pl
import PIL.Image as im
import pyfits 	as pf
import random	as rd

#========================== VARIABLES ==========================#
DIAMETER	= 38.05						# face diameter  (mm)
RIMPAD		= 2.9						# face to arch   (mm)
RADIUS		= DIAMETER / 2					# face radius    (mm)
HEIGHT		= np.sqrt(2) * RADIUS + RIMPAD			# face-to-vertex (mm)
N_ENV		= 1.00000					# environment refractive index
N_CCR		= 1.45702					# CCR refractive index at desired frequency (here fused silica at 532 nm)
DN_CCR		= 10.0e-6					# CCR refractive index temperature dependence
SIZE		= 128						# linear pixel size of diffraction patterns
LOD		= 14						# angular resolution (lambda / D)
TIR		= True						# flag to enable TIR for uncoated CCR
WAVELENGTH	= 632.8e-6					# wavelength of light (mm)
ORIENTATION	= 0.						# CCW rotation of CCR in global frame (radians)
#===============================================================#

#========================== CONSTANTS ==========================#
SENSE		= {'LH':1, 'RH':-1}
SENSEREV	= {1:'LH', -1:'RH'}
PI		= 3.141592653589793
#===============================================================#

class Surface:
	'''The Surface class is instantiated to represent a 
	plane in R3 that contains (0, 0, zheight). Planes can be 
	rotated about the z-axis by providing an orientation 
	angle.'''	
	def __init__(self, dir, zheight):
		self.dir = norm(np.array(dir, dtype='d'))
		self.pos = np.array([0, 0, zheight], dtype='d')
		self.dir = rotate(self.dir, ORIENTATION)
   
class Ray:
	'''The Ray class is instantiated for both the 
	polarization and diffraction pattern ray traces, states 
	being distinguished by inclusion of a distinct path, via
	sequence of surfaces.'''
	def __init__(self, k, b, f, d, path = None, pos = None):
		self.dir		= k
		self.b			= b
		self.f			= f
		self.d			= d
		self.path		= path
		self.surfs_hit		= list()
		self.num_waves		= 0

		# position for diffraction raytraces
		if pos != None: self.get_position(pos)

		# polarization state for polarization raytraces
		if self.path != None: self.get_polarization()
	
	def get_position(self, (i, j, u, v)):
		'''Determines the ray's initial position in the 
		u-v wavefront, a square grid which includes 
		(0, 0, height) as its center.'''
		unit   = DIAMETER / SIZE
		offset = np.array([0, 0, HEIGHT], dtype='d')
		self.pos = (i * unit - RADIUS) * u + (j * unit - RADIUS) * v + offset
	
	def get_polarization(self):
		'''Determine electric field vector parameters 
		described by the current polarization ellipse.
		Initialize parameters in the major-axis frame, 
		then rotate to proper frame.'''
		# normalize amplitudes
		self.amp_s	=     1. / np.sqrt(1 + self.b**2)
		self.amp_p	= self.b / np.sqrt(1 + self.b**2)

		# check if ellipse is linear
		if self.amp_p == 0:
			self.phs_s = 0
			self.phs_p = 0
		else:
			self.phs_s = 0
			self.phs_p = self.d * PI/2

		# rotate ellipse to get
		self.rotate_polarization(-self.f) 		# negated for rotation scheme inversion

	def get_waves(self, p1, p2, dTr, dTz):
		d = dist(p1, p2)
		x1, y1, z1 = p1
		x2, y2, z2 = p2
		
		# return 0 if above corner cube
		if abs(z2) >= HEIGHT and abs(z1) >= HEIGHT: return 0

		# get average depth
		zavg = ( z2 + z1 ) / 2
		
		# get average radius
		if dTr == 0:
			ravg = 0
		else:
			# prepare quadratic coefficients
			c = (x2 - x1)**2 + (y2 - y1)**2
			a = x1**2 + y1**2
			sqrtc = np.sqrt(c)
			sqrta = np.sqrt(a)
			if round(c, 20) == 0: ravg = sqrta
			else:
				a = x1**2 + y1**2
				b = 2 * (x1 * (x2 - x1) + y1 * (y2 - y1))
				
				# computation shortcuts
				sqrtabc = np.sqrt(a + b + c)
				sqrt2   = np.sqrt(2)

				# check discriminant
				disc = b**2 - 4 * a * c
				if round(disc, 8) == 0:
					log = 0
				else:
					num = 2 * sqrtc * sqrtabc + b + 2 * c
					den = 2 * sqrta * sqrtc + b
					if round(den, 10) == 0:
						log = disc = 0
					else:
						log = cmath.log(num) - cmath.log(den)

				ravg = (2 * sqrtc * (b + 2 * c) * sqrtabc - 2 * sqrta * sqrtc * b - disc * log) / (8 * c**1.5)
				ravg = ravg.real

		# get dTavg
		dTavg = dTz * zavg + dTr * ravg		# Eq. 4 in P2

		# get waves
		return DN_CCR * dTavg * d / WAVELENGTH

	def get_ellipse(self):
		'''Determine polarization ellipse parameters 
		described by the current electric field 
		vector.'''
		# get electric field vector location
		dif = self.phs_p - self.phs_s
		es = self.amp_s
		ep = self.amp_p
		sin2dif = np.sin(2 * dif)
		cos2dif = np.cos(2 * dif)
		wt = np.arctan(-ep**2 * sin2dif / (es**2 + ep**2 * cos2dif)) / 2	# equation (16)

		# ellipse vertices
		x1 = es * np.cos(wt)					# equation (17)
		y1 = ep * np.cos(wt + dif)
		x2 = es * np.cos(wt + PI/2)
		y2 = ep * np.cos(wt + PI/2 + dif)
		
		# ellipse axes
		ax1 = np.sqrt(x1**2 + y1**2)
		ax2 = np.sqrt(x2**2 + y2**2)
		
		# ellipse minor axis ratio and angle
		if ax1 > ax2:
			self.f = np.arctan(y1 / x1)
			self.b = ax2 / ax1
		else:
			self.f = np.arctan(y2 / x2)
			self.b = ax1 / ax2

		# ellipse direction
		if   dif >=  PI: dif -= 2 * PI
		elif dif <  -PI: dif += 2 * PI
		self.d = np.sign(dif)

	def get_dist(self, surf):
		'''Return the distance along a ray's direction to
		the given surface.'''
		num = np.dot(surf.pos - self.pos, surf.dir)
		den = np.dot(self.dir, surf.dir)
		return num / den
		
	def rotate_polarization(self, t):
		'''Rotate the electric field vector from one 
		frame to another, according to their angular 
		difference.'''
		coss = np.cos(self.phs_s)
		sins = np.sin(self.phs_s)
		cosp = np.cos(self.phs_p)
		sinp = np.sin(self.phs_p)
		cost = np.cos(t)
		sint = np.sin(t)
		es = self.amp_s
		ep = self.amp_p
		
		# separated trigonometric expansion
		es_coss =  es * coss * cost + ep * cosp * sint		# equations (6) & (7)
		es_sins =  es * sins * cost + ep * sinp * sint
		ep_cosp = -es * coss * sint + ep * cosp * cost
		ep_sinp = -es * sins * sint + ep * sinp * cost

		# compute new phases
		self.phs_s = math.atan2(es_sins, es_coss)	# equation(8)
		self.phs_p = math.atan2(ep_sinp, ep_cosp)
		
		# compute new amplitudes
		mix = 2 * es * ep * cost * sint * (coss * cosp + sins * sinp)
		self.amp_s = cmath.sqrt((es * cost)**2 + (ep * sint)**2 + mix).real
		self.amp_p = cmath.sqrt((es * sint)**2 + (ep * cost)**2 - mix).real

	def refract(self, surf, n1, n2):
		'''Refract the ray from medium 1 (n1) into medium
		2 (n2).  If the ray is being used to trace 
		polarization states, the component amplitudes 
		will decrease according to reflection losses.'''
		# orient surface normal with ray direction
		N = norm(np.dot(self.dir, surf.dir) * surf.dir)
		
		# get angles
		angle = np.dot(self.dir, N)
		if (angle >= 1 or angle <= 1):
			t1 = 0
		else:
			t1    = np.arccos(np.dot(self.dir, N))		# angle of incidence
		t2    = np.arcsin(n1 * np.sin(t1) / n2)		# angle of excitance

		# refract direction
		self.dir = ((n2 * np.cos(t2) - n1 * np.cos(t1)) * N + n1 * self.dir) / n2

		# amplitude losses for polarization raytraces
		if self.path != None:
			n = n2 / n1
			if   t1 != 0:
				self.amp_s *= 1 - (np.sin(t1 - t2) / np.sin(t1 + t2))**2
				self.amp_p *= 1 - (np.tan(t1 - t2) / np.tan(t1 + t2))**2
			else:
				self.amp_s *= (4 * n) / (n + 1)**2
				self.amp_p *= (4 * n) / (n + 1)**2

			# renormalize
			amp = np.sqrt(self.amp_s**2 + self.amp_p**2)
			self.amp_s /= amp
			self.amp_p /= amp

	def reflect(self, surf, n1, n2, hor_axis = None, TIR = None):
		'''Reflect the ray within medium 1 (n1) off 
		medium 2 (n2). If the ray is being used to trace
		polarization states, the component phases will 
		change.'''
		if self.path != None:
			# s, p frame vectors
			s = norm(np.cross(self.dir, surf.dir))			# Eq. (3) in P1
			p = norm(np.cross(s, self.dir))
		
			# rotate to s-p frame
			ver_axis = np.cross(hor_axis, self.dir)
			t = math.atan2(np.dot(s, ver_axis), np.dot(s, hor_axis))	# Eq. (4) in P1
			self.rotate_polarization(t)

			# reflect direction
			self.dir -= 2 * np.dot(surf.dir, self.dir) * surf.dir

			# add phase delay
			t = np.arccos(np.dot(self.dir, surf.dir))
			if TIR:							# Eq. (9), (10) in P1
				sint = np.sin(t)
				cost = np.cos(t)
				rad = np.sqrt((n1 * sint)**2 - 1)
				self.phs_s += 2 * np.arctan(rad / n1 / cost)
				self.phs_p += 2 * np.arctan(rad * n1 / cost)
			else:
				self.phs_s += PI

			return s
		else:
			# reflect direction
			self.dir -= 2 * np.dot(surf.dir, self.dir) * surf.dir

	def hits_front(self, surf, dTr, dTz):
		'''Propogate the ray toward the front face of the
		cornercube.  Return true if the ray hits.'''
		init_pos  = cp.copy(self.pos)
		self.pos += self.dir * self.get_dist(surf)
		self.num_waves += self.get_waves(init_pos, self.pos, dTr, dTz)
		if np.sqrt(self.pos[0]**2 + self.pos[1]**2) <= RADIUS: return True
		return False

	def collide(self, surfaces, dTr, dTz):
		'''Propogate the ray toward the nearest rear 
		surface of the cornercube. Ignore cases where ray
		is already on a surface.  Return nearest 
		surface.'''
		# get distances
		distances = list()
		for surf in surfaces:
			if surf in self.surfs_hit: continue
			distance = self.get_dist(surf)
			distances.append((distance, surf))
		
		# get first minimum distance
		d, surf = sorted(distances)[0]
		
		# propogate to collision
		init_pos  = cp.copy(self.pos)
		self.pos += d * self.dir
		self.num_waves += self.get_waves(init_pos, self.pos, dTr, dTz)
		
		# track collision order
		self.surfs_hit.insert(0,surf)
		
		return surf

def dist(p1, p2):
	'''Return distance between points.'''
	return mag(p2 - p1)

def mag(vec):
	'''Return magnitude of a vector.'''
	return np.sqrt(np.dot(vec, vec))
	
def norm(vec):
	'''Return normalized vector.'''
	return vec / mag(vec)

def radians(deg):
	'''Convert degrees to radians.'''
	return deg * PI / 180
	
def degrees(rad):
	'''Convert radians to degrees.'''
	return rad * 180 / PI
	
def rotate(vec, azi, inc = 0):
	'''Return a vector which has been rotated about the
	Y and Z axes, respectively.'''
	vec  = np.array(vec, dtype='d')
	cinc = np.cos(inc)
	sinc = np.sin(inc)
	cazi = np.cos(azi)
	sazi = np.sin(azi)
	
	# y-axis Rotation Matrix
	R_y = np.array([\
		[ cinc, 0, sinc], \
		[    0, 1,    0], \
		[-sinc, 0, cinc]], dtype='d')
	
	# z-axis Rotation Matrix
	R_z = np.array([\
		[ cazi, -sazi, 0], \
		[ sazi,  cazi, 0], \
		[    0,     0, 1]], dtype='d')

	return np.dot(R_z, np.dot(R_y, vec))

def get_basis(k):
	'''Return basis vectors for wavefront which propogates in
	the k direction.'''
	tangent = np.array([1, 0, 0], dtype='d')
	u = norm(np.cross(tangent, k))
	v = norm(np.cross(k, u))
	return u, v
	
def setup_plot(b, f, d, azimuth):
	'''Return the polarization plot diagram with the input
	polarization state in the lower-lefthand corner.'''
	# draw cornercube and edges
	ax = pl.subplot(111, aspect='equal')
	draw_ellipse(4, 4, 0, 0, 0, ax, 'k')
	k = 'k'
	for i in range (0, 6):
		ax.plot([4.0 * np.cos(i * PI/3 + ORIENTATION), 0], [4.0 * np.sin(i * PI/3 + ORIENTATION), 0], k, lw=.5)
		if   k == 'k:': k = 'k'
		elif k == 'k' : k = 'k:'
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)

	# draw input polarization
	x0 = y0 = -3.33
	b  = b / 2
	f  += azimuth + PI/2
	ax, minor = draw_ellipse(.5, b, f, x0, y0, ax, 'b')

	# add arrows to input polarization
	minor = PI/4
	x1,  y1  = ellipse(.5, b, f, minor)
	x1d, y1d = ellipse(.5, b, f, minor - .001 * d)
	x1,  x1d = x1 + x0, x1d + x0
	y1,  y1d = y1 + y0, y1d + y0
	if b > 0.01:
		pl.arrow(x1, y1, x1d - x1, y1d - y1, width=0.0, head_width=.15, head_length=.15, ec='none', fc='b')
	else:
		xf = 0.4 * np.cos(f)
		yf = 0.4 * np.sin(f)
		pl.arrow(x0, y0, xf,  yf, width=0.0, head_width=0.15, head_length=0.15, ec='none', fc='b')
		pl.arrow(x0, y0,-xf, -yf, width=0.0, head_width=0.15, head_length=0.15, ec='none', fc='b')

	# add directional indicator to ellipse
	ms = 0.05 # x marker size
	ax.plot([x0 - ms, x0 + ms], [y0 - ms, y0 + ms], color='w', lw=3.50) # / border
	ax.plot([x0 - ms, x0 + ms], [y0 + ms, y0 - ms], color='w', lw=3.50) # \ border
	ax.plot([x0 - ms, x0 + ms], [y0 - ms, y0 + ms], color='k', lw=1.50) # / marker
	ax.plot([x0 - ms, x0 + ms], [y0 + ms, y0 - ms], color='k', lw=1.50) # \ marker

	return ax

def add_to_plot(ax, num, azimuth, ray):
	'''Add a ray's polarization ellipse to the polarization
	plot.'''
	# get ellipse
	thet = PI/6 * (1 + 2 * num + 2 * ORIENTATION)
	x0, y0 = -2.5 * np.cos(thet), -2.5 * np.sin(thet)
	ray.f += azimuth + PI/2
	ax, minor = draw_ellipse(1, ray.b, ray.f, x0, y0, ax, 'r')

	# add arrows
	x1,  y1  = ellipse(1, ray.b, ray.f, minor - .200 * ray.d)
	x1d, y1d = ellipse(1, ray.b, ray.f, minor - .201 * ray.d)
	x1,  x1d = x1 + x0, x1d + x0
	y1,  y1d = y1 + y0, y1d + y0
	if ray.b > 0.01:
		pl.arrow(x1, y1, x1d - x1, y1d - y1, width=0.0, head_width=0.25, head_length=0.25 , ec='none', fc='r')
	else:
		xf = 0.8 * np.cos(ray.f)
		yf = 0.8 * np.sin(ray.f)
		pl.arrow(x0, y0, xf,  yf, width=0.0, head_width=0.25, head_length=0.25, ec='none', fc='r')
		pl.arrow(x0, y0,-xf, -yf, width=0.0, head_width=0.25, head_length=0.25, ec='none', fc='r')

	# add directional indicator to ellipse
	pl.plot([x0], [y0], 'w.', markersize=14)
	pl.plot([x0], [y0], 'k.', markersize=9)

def draw_ellipse(a, b, f, x0, y0, ax, c):
	'''Return plot of an ellipse with center (x0, y0), major
	axis a, minor axis b, and major axis-angle f, using color c.'''
	t    = np.linspace(0, 2 * PI, 100)
	x, y = ellipse(a, b, f, t)
	r    = np.sqrt(x**2 + y**2)
	minor = PI/4
	minors = t[np.where(r == r.min())]
	if len(minors) == 1: minors = np.array([minors[0], minors[0] + PI], dtype='d')
	if len(minors) == 2:
		r1 = np.sqrt(sum(np.array(ellipse(a, b, f, minors[0]) - np.array([x0, y0]), dtype='d')**2))
		r2 = np.sqrt(sum(np.array(ellipse(a, b, f, minors[1]) - np.array([x0, y0]), dtype='d')**2))
		if r1 > r2: minor = minors[0]
		else: minor = minors[1]
	x += x0
	y += y0
	ax.plot(x, y, c, lw=1.5)
	return ax, minor

def ellipse(a, b, f, t):
	'''Return (x, y) coordinate of an ellipse with major axis
	a, minor axis b, and axis-angle f for a parameterized
	variable t.'''
	x = a * np.cos(t - f) * np.cos(f) - b * np.sin(t - f) * np.sin(f)
	y = b * np.sin(t - f) * np.cos(f) + a * np.cos(t - f) * np.sin(f)
	return x, y

def make_image(pixels, name, type, directory, invert = False, linearize = False):
	'''Create a grayscale image from an array of values. The
	type flag is used to set the type of images generated.
	The invert flag is used	to invert the grayscale.  For 
	example, inverted .PNG and .FITS images:

	make_image(array, 'filename', 'fp', 1)'''
	# invert
	if invert == 1: pixels = pixels.max() - pixels

	# make fits
	if 'f' in type:
		if name + '.fits' in os.listdir(directory):
			os.unlink(os.path.join(directory, name + '.fits'))
		hdu = pf.PrimaryHDU(np.flipud(pixels))
		hdulist = pf.HDUList([hdu])
		
		hdulist.writeto(os.path.join(directory, name + '.fits'), clobber=True)

	# scale to 0-255
	pixels -= pixels.min()
	m = pixels.max()
	if m != 0:
		pixels *= 255 / m
	
	# render in linear space
	if linearize == 1:
		pixels /= 255
		pixels = 1 - pixels
		pixels = np.power(pixels, 1/2.2)
		pixels = 1 - pixels
		pixels *= 255

	# make rest
	image = im.fromarray(np.uint8(pixels))
	if 'g' in type:
		image.save(os.path.join(directory, name + '.gif'))
	if 'p' in type:
		image.save(os.path.join(directory, name + '.png'))
	if 'e' in type:
		image.save(os.path.join(directory, name + '.eps'))

def diffract(ap, wf):
	'''Returns a diffraction array from an aperture array and
	a wavefront array.  "LOD" stands for lambda / D, denoting
	angular resolution of the diffraction pattern.'''
	# set up enlarge wavefront and aperture arrays
	sbig  = SIZE * LOD
	half  = (sbig - SIZE) / 2
	wfbig = np.zeros((sbig, sbig), dtype='d')
	apbig = np.zeros((sbig, sbig), dtype='d')
	wfbig[half:half + SIZE, half:half + SIZE] = wf
	apbig[half:half + SIZE, half:half + SIZE] = ap
  
	# fast fourier transform
	ft     = np.fft.fft2(apbig * np.exp(-wfbig * 1j))	# the DFT conventions employed by numpy.fft require our argument to be negative.
								# (http://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details)
								# minus sign is equivalent to rotating resulting diffraction pattern by 180 deg
								# can check correct behavior by preparing wavefront tilted by, say, lambda/D
								# and make sure resulting spot in diffrac. pat. shows up in expected quadrant
								# negative phase is interpreted as wavefront delay according to Eq.5 in P1

	powft  = (ft.conj() * ft).real				# equation (18)

	# sort
	sorted = np.zeros((sbig, sbig), dtype='d')
	sorted[:sbig / 2,  :sbig / 2 ] = powft[ sbig / 2:,  sbig / 2:]
	sorted[:sbig / 2,   sbig / 2:] = powft[ sbig / 2:, :sbig / 2 ]
	sorted[ sbig / 2:, :sbig / 2 ] = powft[:sbig / 2,   sbig / 2:]
	sorted[ sbig / 2:,  sbig / 2:] = powft[:sbig / 2,  :sbig / 2 ]

	return sorted[half:half + SIZE, half:half + SIZE]
	
def flux(dif, ctr, rad):
	'''Determine the flux of a diffraction array within a 
	circle centered at a point.  Uses a linear approximation
	for pixels bisected by the circle.'''
	f = 0.0
	xp = np.outer(np.ones(10, dtype='d'), np.linspace(-0.45, 0.45, 10))
	yp = np.outer(np.linspace(-0.45, 0.45, 10), np.ones(10, dtype='d'))
	for x in range(len(dif)):
		for y in range(len(dif[0])):
			r = np.sqrt((x - ctr[0])**2 + (y - ctr[1])**2)
			if (r - rad) < 1.0:
				xgrid = x + xp - ctr[0]
				ygrid = y + yp - ctr[1]
				rgrid = np.sqrt(xgrid * xgrid + ygrid * ygrid)
				whin  = np.where(rgrid < rad)
				count = len(whin[0])
				f    += dif[x][y] * count / 100.0
	return f
