#Title: Pole-Zero map to Bode plot GUI
#made by Doron Shpigel - doron.shpigel@gmail.com
#2024-08-26
#This code is a simple GUI for moving poles and zeros in a pole-zero map and seeing the effect on the bode plot


# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon
from matplotlib.widgets import TextBox
from scipy import signal
from scipy.interpolate import interp1d
import sympy as sp
from sympy import init_printing
init_printing() 




def initialize_array_with_value(Xmin, Xmax, n, X0):
    # Create an array containing n equally spaced numbers between Xmin and Xmax
    arr = np.linspace(Xmin, Xmax, n)
    
    # Check if X0 is already in the array
    if X0 in arr:
        return arr,  np.where(arr == X0)[0][0]
    
    # If X0 is not in the array, find the closest element in the array to X0
    closest_idx = np.abs(arr - X0).argmin()
    
    # Replace the closest element with X0
    arr[closest_idx] = X0
    
    return arr, closest_idx

def enterINT(string):
    print(string)
    while True:
        try:
            return int(input())
        except ValueError:
            print("Invalid input, enter an integer")

drawArea = 14# enterINT("Enter the draw area: ")
GAIN = 1# enterINT("Enter the gain: ")
nDoublePole = 10# enterINT("Enter the number of double poles: ")
nDoubleZero = 10# enterINT("Enter the number of double zeros: ")
nSinglePole = 10# enterINT("Enter the number of single poles: ")
nSingleZero = 10# enterINT("Enter the number of single zeros: ")




def sum_curves(curves):
    # Find the common x-coordinate range
    x_min = min(min(curve[0]) for curve in curves)
    x_max = max(max(curve[0]) for curve in curves)
    x_common = np.linspace(x_min, x_max, 1000)  # You can adjust the number of points as needed

    # Interpolate each curve onto the common x-coordinate range and sum them up
    sum_curve = np.zeros_like(x_common)
    for x, y in curves:
        f = interp1d(x, y, kind='linear')
        y_interp = f(x_common)
        sum_curve += y_interp

    return x_common, sum_curve



#this function is used to update the asymptotic frequency axis, it is auxiliary to the global__asymptotic_frequency_axis function
def update_asymptotic_frequency_axis(frequency_axis, omega):
    max_omega = max(frequency_axis)
    index = np.searchsorted(frequency_axis, omega)
    frequency_axis = np.insert(frequency_axis, index, omega)
    frequency_axis = np.insert(frequency_axis, index, omega)#for the peak beginning
    frequency_axis = np.insert(frequency_axis, index, omega)#for the peak end
    while omega / max_omega <= 1 :
        omega = omega*10
        index = np.searchsorted(frequency_axis, omega)
        frequency_axis = np.insert(frequency_axis, index, omega)
    return frequency_axis

#this function is used to calculate the asymptotic frequency axis, it's return the asymptotic frequency axis
def global_asymptotic_frequency_axis(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero):
    min_omega = min(frequency_axis)
    max_omega = max(frequency_axis)
    small_omega = 1
    big_omega = 10
    while small_omega / min_omega > 1:
        small_omega = small_omega/10
    while big_omega / max_omega < 1:
        big_omega = big_omega*10
    asymptotic_frequency_axis = np.geomspace(small_omega, big_omega, num = len(frequency_axis))
    for (x,y) in AsymptoticDoublePole:
        omega = np.sqrt(x**2 + y**2)
        asymptotic_frequency_axis = update_asymptotic_frequency_axis(asymptotic_frequency_axis, omega)
    for (x,y) in AsymptoticDoubleZero: 
        omega = np.sqrt(x**2 + y**2)
        asymptotic_frequency_axis = update_asymptotic_frequency_axis(asymptotic_frequency_axis, omega)
    for (x,y) in AsymptoticSinglePole:
        omega = abs(x)
        asymptotic_frequency_axis = update_asymptotic_frequency_axis(asymptotic_frequency_axis, omega)
    for (x,y) in AsymptoticSingleZero:
        omega = abs(x)
        asymptotic_frequency_axis = update_asymptotic_frequency_axis(asymptotic_frequency_axis, omega)
    return asymptotic_frequency_axis




# Draggable point class
class DraggablePoint:
    lock = None  # only one can be animated at a time
    def __init__(self, point, index, Single, Pole):
        self.point = point
        self.press = None
        self.background = None
        self.index = index
        self.Single = Single
        self.Pole = Pole
        self.origin = point.center

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.point.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether the mouse is over us; if so, store some data."""
        if (event.inaxes != self.point.axes
                or DraggablePoint.lock is not None):
            return
        contains, attrd = self.point.contains(event)
        if not contains:
            return
        print('event contains', self.point.center)
        self.press = self.point.center, (event.xdata, event.ydata)
        DraggablePoint.lock = self

        # draw everything but the selected point and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(axes.bbox)

        # now redraw just the point
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Move the point if the mouse is over us."""
        if (event.inaxes != self.point.axes
                or DraggablePoint.lock is not self):
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        x = round(x0 +dx, 1)
        if (self.Single == False):
            y = round(y0 +dy, 1)
        else: y = 0

        if not((x >  - drawArea) and (x < drawArea) and (y > -drawArea) and (y < drawArea)):
            x, y = self.origin

        self.point.set_center((x, y))
        if (self.Single == False) and (self.Pole == True):
            DoubleMirrorPole[self.index].set_center((x, -y))
            print('double pole at x =', x, 'y =', y)
        elif (self.Single == False) and (self.Pole == False):
            DoubleMirrorZero[self.index].set_center((x, -y))
            print('double zero at x =', x, 'y =', y)
        elif (self.Single == True) and (self.Pole == True):
            print('single pole at x =', x, 'y = 0')
        elif (self.Single == True) and (self.Pole == False):
            print('single zero at x =', x, 'y = 0')

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current point
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        """Clear button press information."""
        if DraggablePoint.lock is not self:
            return
        transferfunction()
        self.press = None
        DraggablePoint.lock = None

        # turn off the point animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

# Transfer function creation for printing
def transfer_function_to_sympy(num, den, gain):
    s = sp.symbols('s')
    numerator_poly = 1
    denominator_poly = 1
    if num != []:
        for i in num:
            numerator_poly *= (s - i)
    if den != []:
        for i in den:
            denominator_poly *= (s - i)
    # Create transfer function expression
    transfer_function_expr = gain * numerator_poly / denominator_poly
    return transfer_function_expr
# Transfer function calculation
def transferfunction():
    num = []
    den = []
    AsymptoticDoublePole = []
    AsymptoticDoubleZero = []
    AsymptoticSinglePole = []
    AsymptoticSingleZero = []
    for point in SumPoints:
        x, y = point.center
        if (x >  - drawArea) and (x < drawArea) and (y > -drawArea) and (y < drawArea):
            if (point in DoublePole):
                den.append(x + y*1j)
                den.append(x - y*1j)
                AsymptoticDoublePole.append((x, y))
            elif (point in DoubleZero):
                num.append(x + y*1j)
                num.append(x - y*1j)
                AsymptoticDoubleZero.append((x, y))
            elif (point in SinglePole):
                den.append(x)
                AsymptoticSinglePole.append((x, 0))
            elif (point in SingleZero):
                num.append(x)
                AsymptoticSingleZero.append((x, 0))
    print('the transfer function has num = ', num,' den = ' ,den, 'and gain = ', GAIN)
    s = signal.lti(num, den, GAIN)
    #print("The poles are: ", s.poles)
    #print("The zeros are: ", s.zeros)
    frequency_axis, mag, phase = signal.bode(s, n = 1000)
    mag_plot(plotMagnitude, frequency_axis, mag)
    phase_plot(plotPhase, frequency_axis, phase)
    asymptoticPlot(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero)
    #print('The transfer function is:', transfer_function_to_sympy(num, den, GAIN))
    sp.latex
    fig.suptitle('H(s) = '+ str(transfer_function_to_sympy(num, den, GAIN)), fontsize=16)

def asymptoticMagnitudeArray(slope, frequency_axis, omega, type, xi=1):
    magnitude = np.zeros(len(frequency_axis))
    omega_index = np.searchsorted(frequency_axis, omega)
    #handle the peak case:
    peak_index = omega_index + 1
    if type == 'double-pole':
        xi = abs(xi)*(-1)
    elif type == 'double-zero':
        xi = abs(xi)
    #handle the peak case:
    if xi > -0.5 and xi < 0.5:
        magnitude[peak_index] = np.sign(xi)*20*np.log10(2*np.abs(xi)) #negative sign for poles, positive for zeros
    else:
        magnitude[peak_index] = 0
    peak_index += 1 # for the peak ending
    magnitude[peak_index] = 0 
    #make linear slope
    x1 = np.searchsorted(frequency_axis, omega, side='right')
    x2 = np.searchsorted(frequency_axis, omega*10, side='left')
    m = slope/np.log10(frequency_axis[x2]/frequency_axis[x1])
    for i in range(x1, len(frequency_axis)):
        magnitude[i] = m*np.log10(frequency_axis[i]) -m*np.log10(frequency_axis[x1])
    return magnitude

def asymptoticPhaseArray(w, omega, type, xi=1):
    _omega = omega
    min_w = min(w)
    minimum = min(min_w, _omega*(10**(-abs(xi))))
    w=[minimum, _omega*(10**(-abs(xi)))]
    phase = [0, 0]
    
    if type == 'pole':
        phase.append(-45)
        phase.append(-90)
        phase.append(-90)
        w.append(_omega)
        w.append(_omega*10**(abs(xi)))
        w.append(_omega*10**(2*abs(xi)))
    elif type == 'zero':
        phase.append(45)
        phase.append(90)
        phase.append(90)
        w.append(_omega)
        w.append(_omega*10**(abs(xi)))
        w.append(_omega*10**(2*abs(xi)))
    elif type == 'double-pole':
        phase.append(-90)
        phase.append(-180)
        phase.append(-180)
        w.append(_omega)
        w.append(_omega*10**(abs(xi)))
        w.append(_omega*10**(2*abs(xi)))
    elif type == 'double-zero':
        phase.append(90)
        phase.append(180)
        phase.append(180)
        w.append(_omega)
        w.append(_omega*10**(abs(xi)))
        w.append(_omega*10**(2*abs(xi)))
    return w, phase

def Asymptotic_frequency_axis(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero):
    min_omega = min(frequency_axis)
    max_omega = max(frequency_axis)
    for (x,y) in AsymptoticDoublePole:
        omega = np.sqrt(x**2 + y**2)
        if omega > max_omega:
            max_omega = omega
        if omega < min_omega:
            min_omega = omega
    for (x,y) in AsymptoticDoubleZero:
        omega = np.sqrt(x**2 + y**2)
        if omega > max_omega:
            max_omega = omega
        if omega < min_omega:
            min_omega = omega
    for (x,y) in AsymptoticSinglePole:
        omega = abs(x)
        if omega > max_omega:
            max_omega = omega
        if omega < min_omega:
            min_omega = omega
    for (x,y) in AsymptoticSingleZero:
        omega = abs(x)
        if omega > max_omega:
            max_omega = omega
        if omega < min_omega:
            min_omega = omega
    omega=max_omega
    max_omega_value = max_omega
    min_omega_value = min_omega
    dec = 0
    base = 10
    while int(omega) > 0:
        omega = omega/10
        dec += 1
    max_omega_dec = base**dec
    while base> min_omega:
        base = base/10
    min_omega_dec = base
    asymptotic_frequency_axis = np.geomspace(min_omega_dec, max_omega_dec, num = len(frequency_axis))
    return asymptotic_frequency_axis
# Asymptotic plot calculation
#TODO: make AsymptoticOriginPole
#TODO: make AsymptoticOriginZero
#TODO: make make sum curve for phase
def asymptoticPlot(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero):
    plotMagnitudeAsymptotic.clear()
    plotMagnitudeAsymptotic.grid(True)
    plotPhaseAsymptotic.clear()
    plotPhaseAsymptotic.grid(True)
    #place holders for the contribution of each calculation
    constant = 1
    asymptotic_frequency_axis = global_asymptotic_frequency_axis(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero)
    sum_curve = np.zeros(len(asymptotic_frequency_axis))
    for (x,y) in AsymptoticDoublePole:
        omega = np.sqrt(x**2 + y**2)
        xi = abs(x)/omega
        constant *= 1/(omega**2)
        slope = -40
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'double-pole', xi)
        sum_curve = np.add(sum_curve, magnitude) #summing the contribution of each pole
        plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='red', label= f"double pole at {x,y}")
        W, phase = asymptoticPhaseArray(frequency_axis, omega, 'double-pole', xi)
        plotPhaseAsymptotic.semilogx(W, phase, color='red', label= f"double pole at {x,y}")
    for (x,y) in AsymptoticDoubleZero:
        omega = np.sqrt(x**2 + y**2)
        xi = abs(x)/omega
        constant *= omega**2
        slope = 40
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'double-zero', xi)
        sum_curve = np.add(sum_curve, magnitude) #summing the contribution of each pole
        plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='blue', label= f"double zero at {x,y}")
        W, phase = asymptoticPhaseArray(frequency_axis, omega, 'double-zero', xi)
        plotPhaseAsymptotic.semilogx(W, phase, color='blue', label= f"double zero at {x,y}")
    for (x,y) in AsymptoticSinglePole:
        omega = abs(x)
        constant *= 1/omega
        slope = -20
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'pole')
        sum_curve = np.add(sum_curve, magnitude) #summing the contribution of each pole
        plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='pink', label= f"single pole at {x,y}")
        W, phase = asymptoticPhaseArray(frequency_axis, omega, 'pole')
        plotPhaseAsymptotic.semilogx(W, phase, color='pink', label= f"single pole at {x,y}")
    for (x,y) in AsymptoticSingleZero:
        omega = abs(x)
        constant *= omega
        slope = 20
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'zero')
        sum_curve = np.add(sum_curve, magnitude) #summing the contribution of each pole
        plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='cyan', label= f"single zero at {x,y}")
        W, phase = asymptoticPhaseArray(frequency_axis, omega, 'zero')
        plotPhaseAsymptotic.semilogx(W, phase, color='cyan', label= f"single zero at {x,y}")
    #constant K calculation
    K = GAIN*constant
    W = asymptotic_frequency_axis
    Kmagnitude = 20*np.log10(abs(K))*np.ones(len(W))
    #interpolating the magnitude to the common x-axis
    # f = interp1d(W, Kmagnitude, kind='nearest')
    # sum_curve += f(np.clip(asymptotic_frequency_axis, min(W), max(W)))
    sum_curve = np.add(sum_curve, Kmagnitude)
    Kphase=(1-np.sign(K))*90*np.ones(len(W))
    #plotting constant K
    plotMagnitudeAsymptotic.semilogx(W, Kmagnitude, color='green', label=f'constant K={round(K,2)}')
    plotPhaseAsymptotic.semilogx(W, Kphase, color='green', label=f'constant K={round(K,2)}')
    #plotting the sum of all contributions
    plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, sum_curve, color='black', label='sum')
    #adding legend
    plotMagnitudeAsymptotic.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plotPhaseAsymptotic.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plotMagnitudeAsymptotic.set_title('Asymptotic Bode Plot - Magnitude plot')
    plotPhaseAsymptotic.set_title('Asymptotic Bode Plot - Phase plot')

# Pole-Zero map subplot configuration
def pole_zero_map(ax):
    for i in range(-drawArea, drawArea+1):
        ax.axvline(i, color='gray', linestyle='dashed', linewidth=0.5)
        ax.axhline(i, visible = False, color='gray', linestyle='dashed', linewidth=0.5)
        ax.hlines(i, -drawArea, drawArea, color='gray', linestyle='dashed', linewidth=0.5)
    ax.arrow(-drawArea, 0, 2*drawArea, 0, head_width=0.5, head_length=0.5, fc='grey', ec='black', label='Real')
    ax.arrow(0, -drawArea, 0, 2*drawArea, head_width=0.5, head_length=0.5, fc='grey', ec='black', label='Imaginary')
    ticktoremove = np.arange(-drawArea-10, -drawArea, 1)
    def remove_tick(x, pos):
        if x < -drawArea:
            return ''
        else:
            return str(round(x,1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(remove_tick))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(remove_tick))
    recBlank = Rectangle((-drawArea, -drawArea), 2*drawArea, 2*drawArea, fill = False, edgecolor='black')
    recSingle = Rectangle((-1*drawArea-10, -0.75), 5, 1.5, fill=False)
    recDouble = Rectangle((-1*drawArea-5, -2.5), 5, 5, fill=False)
    ax.add_patch(recBlank)
    ax.add_patch(recDouble)
    ax.add_patch(recSingle)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Pole-Zero Map')
# Bode plot magnitude subplot configuration
def mag_plot(ax, w, mag):
    ax.clear()
    ax.semilogx(w, mag)
    ax.yaxis.tick_right()
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Magnitude (dB)')
    ax.yaxis.set_label_position("right")
    ax.set_title('Bode Plot - Magnitude plot')
    ax.grid(True)
# Bode plot phase subplot configuration
def phase_plot(ax, w, phase):
    ax.clear()
    ax.semilogx(w, phase)
    ax.yaxis.tick_right()
    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Phase (degrees)')
    ax.yaxis.set_label_position("right")
    ax.set_title('Bode Plot - Phase plot')
    ax.grid(True)

# Main function

def changeGain(text):
    global GAIN
    GAIN = float(text)

# Create the figure and subplots
fig = plt.figure(tight_layout=False)
gs = gridspec.GridSpec(2, 3)
leftAX = fig.add_subplot(gs[:, 0])
plotMagnitude = fig.add_subplot(gs[0, 1])
plotPhase = fig.add_subplot(gs[1, 1])
plotMagnitudeAsymptotic = fig.add_subplot(gs[0, 2])
plotPhaseAsymptotic = fig.add_subplot(gs[1, 2])
# Adding TextBox to figure to change the gain
graphBox = fig.add_axes([0.074, 0.06, 0.075, 0.075])
txtBox = TextBox(graphBox, "Gain: ")
txtBox.on_submit(changeGain)
txtBox.set_val("1")
# Arrays for the poles and zeros
DoublePole = [Circle((-1*drawArea-5+1.5, 1), radius=0.5, color='red', fill = True) for _ in range(nDoublePole)]
DoubleMirrorPole = [Circle((-1*drawArea-5+1.5, -1), radius=0.5, color='pink', fill = True) for _ in range(nDoublePole)]
DoubleZero = [Circle((-1*drawArea-5+3.5, 1), radius=0.5, color='blue', fill = False) for _ in range(nDoubleZero)]
DoubleMirrorZero = [Circle((-1*drawArea-5+3.5, -1), radius=0.5, color='cyan', fill = False) for _ in range(nDoubleZero)]
SinglePole = [Circle((-1*drawArea-10+1.5, 0), radius=0.5, color='red', fill = True) for _ in range(nSinglePole)]
#SingleZero = [Wedge((-1*drawArea-10+3.5, 0), 0.75, 200, 340, color='blue', fill = False) for _ in range(nSingleZero)]
SingleZero = [Circle((-1*drawArea-10+3.5, 0), radius=0.5, color='blue', fill = False) for _ in range(nDoublePole)]
SumPoints = []

# Add the patches to the left subplot
for point in DoublePole:
    leftAX.add_patch(point)
    leftAX.add_patch(DoubleMirrorPole[DoublePole.index(point)])
for point in DoubleZero:
    leftAX.add_patch(point)
    leftAX.add_patch(DoubleMirrorZero[DoubleZero.index(point)])
for point in SinglePole:
    leftAX.add_patch(point)
for point in SingleZero:
    leftAX.add_patch(point)
# setting the pole-zero map subplot
pole_zero_map(leftAX)

# Create the draggable points
drs = []
for point in DoublePole:
    SumPoints.append(point)
    index = DoublePole.index(point)
    dr = DraggablePoint(point, index, False, True)
    dr.connect()
    drs.append(dr)
for point in DoubleZero:
    SumPoints.append(point)
    index = DoubleZero.index(point)
    dr = DraggablePoint(point, index, False, False)
    dr.connect()
    drs.append(dr)
for point in SinglePole:
    SumPoints.append(point)
    index = SinglePole.index(point)
    dr = DraggablePoint(point, index, True, True)
    dr.connect()
    drs.append(dr)
for point in SingleZero:
    SumPoints.append(point)
    index = SingleZero.index(point)
    dr = DraggablePoint(point, index, True, False)
    dr.connect()
    drs.append(dr)

# running the GUI
plt.show()

