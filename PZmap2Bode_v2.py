#Title: Pole-Zero map to Bode plot GUI
#made by Doron Shpigel - doron.shpigel@gmail.com
#2024-08-26
#This code is a simple GUI for moving poles and zeros in a pole-zero map and seeing the effect on the bode plot

#TODO: make AsymptoticOriginPole
#TODO: make AsymptoticOriginZero

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

print_points_in_terminal = False
pick_radius_for_legend = 5


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



class Sum_Curve:
    def __init__(self, frequency_axis):
        self.frequency_axis = frequency_axis
        self.magnitude = np.zeros(len(frequency_axis))
        self.phase = np.zeros(len(frequency_axis))
        self.constant = 1
    def add_double_pole(self, x, y):
        omega = np.sqrt(x**2 + y**2)
        self.constant *= 1/(omega**2)
    def add_double_zero(self, x, y):
        omega = np.sqrt(x**2 + y**2)
        self.constant *= omega**2
    def add_single_pole(self, x, y):
        omega = abs(x)
        self.constant *= 1/omega
    def add_single_zero(self, x, y):
        omega = abs(x)
        self.constant *= omega
    #constant K calculation
    def calculate_constant(self):
        self.K = GAIN*self.constant
        self.Kmagnitude = 20*np.log10(abs(self.K))*np.ones(len(self.frequency_axis))
        self.Kphase=(1-np.sign(self.K))*90*np.ones(len(self.frequency_axis))
        return self.K, self.Kmagnitude, self.Kphase
    def add_magnitude(self, magnitude):
        self.magnitude = np.add(self.magnitude, magnitude)
    def add_phase(self, phase):
        self.phase = np.add(self.phase, phase)
    def sum(self):
        self.add_magnitude(self.Kmagnitude)
        self.add_phase(self.Kphase)
        return self.magnitude, self.phase






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
        # print('event contains', self.point.center)
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
            if print_points_in_terminal:
                print('double pole at x =', x, 'y =', y)
        elif (self.Single == False) and (self.Pole == False):
            DoubleMirrorZero[self.index].set_center((x, -y))
            if print_points_in_terminal:
                print('double zero at x =', x, 'y =', y)
        elif (self.Single == True) and (self.Pole == True):
            if print_points_in_terminal:
                print('single pole at x =', x, 'y = 0')
        elif (self.Single == True) and (self.Pole == False):
            if print_points_in_terminal:
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
    peak_index = omega_index + 1 #for the peak beginning
    if type == 'double-pole':
        xi = abs(xi)*(-1)
    elif type == 'double-zero':
        xi = abs(xi)
    #insert the peak value or 0
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

def asymptoticPhaseArray(frequency_axis, omega, type, xi=1):
    phase = np.zeros(len(frequency_axis))
    low_freq_index = np.searchsorted(frequency_axis, omega/10, side='right')
    high_freq_index = np.searchsorted(frequency_axis, omega*10, side='left')
    #for the low frequencies we already defined the phase 0
    for i in range(high_freq_index, len(frequency_axis)):
        if type == 'pole':
            phase[i] = -90
        elif type == 'zero':
            phase[i] = 90
        elif type == 'double-pole':
            phase[i] = -180
        elif type == 'double-zero':
            phase[i] = 180
    #we need to find the linear slope between the low and high frequencies
    slope = phase[high_freq_index] 
    m = slope/np.log10(frequency_axis[high_freq_index]/frequency_axis[low_freq_index])
    for i in range(low_freq_index, high_freq_index):
        phase[i] = m*np.log10(frequency_axis[i]) - m*np.log10(frequency_axis[low_freq_index])
    return phase


# Asymptotic plot calculation
def asymptoticPlot(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero):
    plotMagnitudeAsymptotic.clear()
    plotMagnitudeAsymptotic.grid(True)
    plotPhaseAsymptotic.clear()
    plotPhaseAsymptotic.grid(True)
    #place holders for the contribution of each calculation
    asymptotic_frequency_axis = global_asymptotic_frequency_axis(frequency_axis, AsymptoticDoublePole, AsymptoticDoubleZero, AsymptoticSinglePole, AsymptoticSingleZero)
    summery = Sum_Curve(asymptotic_frequency_axis)
    lines_magnitude=[]
    lines_phase=[]
    for (x,y) in AsymptoticDoublePole:
        summery.add_double_pole(x, y)
        omega = np.sqrt(x**2 + y**2)
        xi = abs(x)/omega
        slope = -40
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'double-pole', xi)
        (line_magnitude, ) = plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='red', label= f"double pole at {x,y}")
        phase = asymptoticPhaseArray(asymptotic_frequency_axis, omega, 'double-pole', xi)
        (line_phase, ) = plotPhaseAsymptotic.semilogx(asymptotic_frequency_axis, phase, color='red', label= f"double pole at {x,y}")
        summery.add_magnitude(magnitude)
        summery.add_phase(phase)
        lines_magnitude.append(line_magnitude)
        lines_phase.append(line_phase)
    for (x,y) in AsymptoticDoubleZero:
        summery.add_double_zero(x, y)
        omega = np.sqrt(x**2 + y**2)
        xi = abs(x)/omega
        slope = 40
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'double-zero', xi)
        (line_magnitude, ) = plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='blue', label= f"double zero at {x,y}")
        phase = asymptoticPhaseArray(asymptotic_frequency_axis, omega, 'double-zero', xi)
        (line_phase, ) = plotPhaseAsymptotic.semilogx(asymptotic_frequency_axis, phase, color='blue', label= f"double zero at {x,y}")
        summery.add_magnitude(magnitude)
        summery.add_phase(phase)
        lines_magnitude.append(line_magnitude)
        lines_phase.append(line_phase)
    for (x,y) in AsymptoticSinglePole:
        summery.add_single_pole(x, y)
        omega = abs(x)
        slope = -20
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'pole')
        (line_magnitude, ) = plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='pink', label= f"single pole at {x,y}")
        phase = asymptoticPhaseArray(asymptotic_frequency_axis, omega, 'pole')
        (line_phase) = plotPhaseAsymptotic.semilogx(asymptotic_frequency_axis, phase, color='pink', label= f"single pole at {x,y}")
        summery.add_magnitude(magnitude)
        summery.add_phase(phase)
        lines_magnitude.append(line_magnitude)
        lines_phase.append(line_phase)
    for (x,y) in AsymptoticSingleZero:
        summery.add_single_zero(x, y)
        omega = abs(x)
        slope = 20
        magnitude= asymptoticMagnitudeArray(slope, asymptotic_frequency_axis, omega, 'zero')
        (line_magnitude, ) = plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='cyan', label= f"single zero at {x,y}")
        phase = asymptoticPhaseArray(asymptotic_frequency_axis, omega, 'zero')
        (line_phase, ) = plotPhaseAsymptotic.semilogx(asymptotic_frequency_axis, phase, color='cyan', label= f"single zero at {x,y}")
        summery.add_magnitude(magnitude)
        summery.add_phase(phase)
        lines_magnitude.append(line_magnitude)
        lines_phase.append(line_phase)
    #constant K calculation
    K, Kmagnitude, Kphase = summery.calculate_constant()
    (line_magnitude, ) = plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, Kmagnitude, color='green', label=f'constant K={round(K,2)}')
    lines_magnitude.append(line_magnitude)
    (line_phase, ) = plotPhaseAsymptotic.semilogx(asymptotic_frequency_axis, Kphase, color='green', label=f'constant K={round(K,2)}')
    lines_phase.append(line_phase)
    # summing all contributions
    magnitude, phase = summery.sum()
    (line_magnitude, ) = plotMagnitudeAsymptotic.semilogx(asymptotic_frequency_axis, magnitude, color='black', label='sum')
    lines_magnitude.append(line_magnitude)
    (line_phase, ) = plotPhaseAsymptotic.semilogx(asymptotic_frequency_axis, phase, color='black', label='sum')
    lines_phase.append(line_phase)
    map_legend_to_ax = {}  # Will map legend lines to original lines.

    #adding legend
    legend_magnitude = plotMagnitudeAsymptotic.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fancybox=True, shadow=True)
    legend_phase = plotPhaseAsymptotic.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plotMagnitudeAsymptotic.set_title('Asymptotic Bode Plot - Magnitude plot')
    plotPhaseAsymptotic.set_title('Asymptotic Bode Plot - Phase plot')

    for legend_line, ax_line in zip(legend_magnitude.get_lines(), lines_magnitude):
        legend_line.set_picker(pick_radius_for_legend)  # Enable picking on the legend line.
        map_legend_to_ax[legend_line] = ax_line
    for legend_line, ax_line in zip(legend_phase.get_lines(), lines_phase):
        legend_line.set_picker(pick_radius_for_legend) # Enable picking on the legend line.
        map_legend_to_ax[legend_line] = ax_line

    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legend_line = event.artist

        # Do nothing if the source of the event is not a legend line.
        if legend_line not in map_legend_to_ax:
            return

        ax_line = map_legend_to_ax[legend_line]
        visible = not ax_line.get_visible()
        ax_line.set_visible(visible)
        # Change the alpha on the line in the legend, so we can see what lines
        # have been toggled.
        legend_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', on_pick)


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

# Function to change the gain in the GUI
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

