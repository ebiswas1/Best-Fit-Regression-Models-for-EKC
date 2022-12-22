import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statistics import mode
from scipy.interpolate import RBFInterpolator


def Method1(gdpo, co2o):
    """Polynomial Regression"""
    #Written by Ben
    #Reformat data for sklearn regression function
    gdp = gdpo[:,np.newaxis]
    co2 = co2o[:,np.newaxis]

    #Perform regression
    polyfeatures = PolynomialFeatures(degree=3)
    feat = polyfeatures.fit_transform(gdp)
    model = LinearRegression()
    model.fit(feat, co2)
    co2poly = model.predict(feat)

    #Calculate R-Squared
    rsq = r2_score(co2,co2poly)

    return model, rsq

def PlotMethod1(model, gdpv):
    """Create GHG Vector to Plot Using Method 1"""
    coeffs = model.coef_[0]
    intercept = model.intercept_[0]
    co2v = coeffs[3]*gdpv**3 + coeffs[2]*gdpv**2 + coeffs[1]*gdpv + coeffs[0] + intercept
    return co2v

def Method2(gdpo,co2o):
    """Polynomial Regression with Time Trend"""
    #Written by Ben
    co2 = co2o[:,np.newaxis]
    lyindex = len(gdpo) - 1
    time = np.linspace(0, lyindex, lyindex + 1)
    gdp2 = np.stack([gdpo,gdpo**2,gdpo**3,time],axis=1) #For this model, I will treat it as a linear regression

    #Perform regression
    polyfeatures = PolynomialFeatures(degree=1)
    feat = polyfeatures.fit_transform(gdp2)
    model = LinearRegression()
    model.fit(feat, co2)
    co2poly = model.predict(feat)

    #Calculate R-Squared
    rsq = r2_score(co2,co2poly)

    return model, rsq

def PlotMethod2(model, gdpv, timev):
    """Create GHG Vector to Plot Using Method 2"""
    #Save coefficients for regression
    coeffs = model.coef_[0]
    intercept = model.intercept_[0]

    #Calculate corresponding co2 output
    co2v = coeffs[3]*gdpv**3 + coeffs[2]*gdpv**2 + coeffs[1]*gdpv + coeffs[0] + coeffs[4]*timev + intercept
    return co2v

def Method3(gdp,co2):
    """Semi-Parametric with Regression Spline"""
    #Written by Eletria

    # Reformat data for sklearn regression function
    gdp = gdpo[:,np.newaxis]
    co2 = co2o[:,np.newaxis]

    # Create time trend for the data
    lyindex = len(gdpo) - 1
    time = np.linspace(0, lyindex, lyindex + 1)
    gdp2 = np.stack([gdpo,gdpo**2,gdpo**3,time],axis=1)

    # Semi parametric regression function
    def semi_parametric_regression(co2, gdp, time):
        ones = np.ones((len(time)))
        timeStack = np.vstack((time,ones))
        # Compute the coefficients for the linear part of the model using ordinary least squares
        B1, B0 = np.linalg.lstsq(timeStack.T, co2, rcond=None)[0]

        # Compute the residuals by subtracting the linear part of the model from the co2 values
        residuals = co2 - (B1 * time + B0)

        # Compute the non-parametric part of the model using the residuals and the gdp values
        f = np.polyfit(gdp, residuals, deg=2)

        # Return the coefficients of the linear part of the model and the polynomial representing the non-parametric part
        return B0, B1, f
    # Assigning coefficients of linear part and second degree polynomial values
    vals=semi_parametric_regression(co2o, gdpo, time)
    B0,B1=vals[:2] # Coefficients
    func=vals[2] # Second degree polynomial function of gdp

    # Align shapes for predicted CO2 values
    x=gdp.T
    gdpm=x[0]

    # Calculate predicted CO2 values
    co2c = B0 + B1*time + func[0]*gdpm**2 + func[1]*gdpm + func[2]

    # Calculate MSE and R-squared
    rsq = r2_score(co2,co2c)

    return vals,rsq

def PlotMethod3(model, gdpv, timev):
    """Create GHG Vector to Plot Using Method 3"""
    # Generate data to plot using smaller step size
    vals = model
    B0,B1=vals[:2]
    func=vals[2]

    co2v = B0 + B1*timev + func[0]*gdpv**2 + func[1]*gdpv + func[2]
    return co2v

def Method4(gdp,co2):
    """Thin Plate Cubic Splines"""
    #Written by Mack
    # setting up inputs
    lyindex = len(gdpo) - 1
    time = np.linspace(0, lyindex, lyindex + 1)
    input1 = np.column_stack((gdp,time)) #needs to be array for input into RBF function

    tps = RBFInterpolator(input1, co2,smoothing=10**10, degree=3) # creating TPS object/function
    predict = tps(input1) #running it on orginal inputs, has to be specific shape
    rsq = r2_score(co2, predict) #comparing model outputs to data
    return tps, rsq

def PlotMethod4(model, gdpv, timev):
    """Create GHG Vector to Plot Using Method 4"""
    input1 = np.column_stack((gdpv,timev))
    co2v = model(input1)
    test = np.argmax(co2v)
    return co2v



#Import Data from Excel Files
co2df = pd.read_excel(r'co2data.xlsx')
gdpdf = pd.read_excel(r'gdpdata.xlsx')
popdf = pd.read_excel(r'popdata.xlsx')

#Establish Empty Arrays to hold data
gdptotal = np.zeros(50)
co2total = np.zeros(50)
pop = np.zeros(50)
co2o = np.zeros(50)

#Produce list of all available countries
adc = []
for country in gdpdf.columns:
    if country in popdf.columns and country in co2df['country'].values:
        adc.append(country)

#Create a list to hold best methods
bestmethods = []
methods = [1,2,3,4]
bestr2 = []

#Establish All GHG/capita as the pollutant to focus on
GHGType = 'GHG' #Possible GHG Types: 'CO2','CH4','N2O','Fgas','GHG'
if GHGType not in ['CO2','CH4','N2O','Fgas','GHG']:
    raise ValueError("This GHG Type is Not in the Emissions Dataset")

#Run all methods on all countries to establish which is best
#R2 used as metric for comparison
for country in adc:
    CountryName = country
    r2v = np.zeros(4)
    modelv = [None]*4

    ##### Populate np arrays with data from appropriate country and pollutant types #####
    if CountryName in co2df['country'].values and CountryName in gdpdf.columns and CountryName in popdf.columns:
        gdpo = np.array(gdpdf[CountryName])
        pop = np.array(popdf[CountryName])
        co2index = co2df.index[co2df['country']==CountryName].tolist()[0]
        co2total = np.array(co2df.loc[co2index:co2index+50][GHGType])
        assert(len(co2total) == len(pop) == len(gdpo))
        co2o = co2total/pop
    else:
        raise ValueError("This Country is Not in All 3 Datasets")


    co2 = co2o.copy()
    gdp = gdpo.copy()

    ##### Run All Methods And Store R2 Value and Model #####
    modelv[0], r2v[0] = Method1(gdpo,co2o)
    modelv[1], r2v[1] = Method2(gdpo,co2o)
    modelv[2], r2v[2] = Method3(gdpo,co2o)
    modelv[3], r2v[3] = Method4(gdpo,co2o)

    ##### Select Best Method #####
    #The paper suggests choosing based off of r2, deviance, and deviance/degrees of freedom
    #For the sake of simplicity here, we will simply be using r2 as their calculations for the other 2 are a bit unclear
    index = np.argmax(r2v)
    bestmethod = methods[index]
    bestmethods.append(bestmethod)
    bestr2.append(r2v[index])

    ##### Plot only the best method #####
    #Save best model
    model = modelv[index]

    #Important Constants
    lyindex = len(gdpo) - 1
    mingdp = min(gdpo)
    maxgdp = max(gdpo)
    steps = lyindex*10 + 1

    #Create gdp and time vector along smaller step size
    gdpv = np.linspace(mingdp,maxgdp,steps)
    timev = np.linspace(0,lyindex,steps)


    if bestmethod == 1:
        co2v = PlotMethod1(model,gdpv)

        #Create and Save Plot Figure
        plt.plot(gdp,co2,"o")
        plt.plot(gdpv,co2v)
        plt.title(f"{CountryName} Plot Using Best Method 1")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Greenhouse Gas Emissions per Capita")
        plt.savefig(f'CountryPlots/{CountryName}.png')
        plt.close()

    elif bestmethod == 2:
        co2v = PlotMethod2(model, gdpv, timev)

        #Create and Save Plot Figure
        plt.plot(gdp,co2,"o")
        plt.plot(gdpv,co2v)
        plt.title(f"{CountryName} Plot Using Best Method 2")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Greenhouse Gas Emissions per Capita")
        plt.savefig(f'CountryPlots/{CountryName}.png')
        plt.close()

    elif bestmethod == 3:
        co2v = PlotMethod3(model, gdpv, timev)

        plt.plot(gdp,co2,"o")
        plt.plot(gdpv,co2v)
        plt.title(f"{CountryName} Plot Using Best Method 4")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Greenhouse Gas Emissions per Capita")
        plt.close()

    elif bestmethod == 4:
        co2v = PlotMethod4(model, gdpv, timev)

        #Create and Save Plot Figure
        plt.plot(gdp,co2,"o")
        plt.plot(gdpv,co2v)
        plt.title(f"{CountryName} Plot Using Best Method 4")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Greenhouse Gas Emissions per Capita")
        plt.savefig(f'CountryPlots/{CountryName}.png')
        plt.close()

    #Plot all methods
    plt.plot(gdp,co2,"o")
    plt.plot(gdpv,PlotMethod1(modelv[0],gdpv),label="Method 1")
    plt.plot(gdpv,PlotMethod2(modelv[1],gdpv,timev),label="Method 2")
    plt.plot(gdpv,PlotMethod3(modelv[2],gdpv,timev),label="Method 3")
    plt.plot(gdpv,PlotMethod4(modelv[3],gdpv,timev),label="Method 4")
    plt.legend()
    plt.title(f"{CountryName} Plot Using All Methods")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Greenhouse Gas Emissions per Capita")
    plt.savefig(f'CountryAllMethodPlots/{CountryName}AllMethods.png')
    plt.close()


#Plot histogram showing distribution into different methods
#Useless because they are all the same
plt.hist(bestr2)
plt.title("Histogram of Best Fit R2 Values")
plt.show()

#Show box and whisker distribution of best fit r2 values. Some countries have very low r2
plt.boxplot(bestr2)
plt.title("Distribution of Best Fit R2 Values")
plt.show()

#Output Data to Excel File
output={
    "CountryName": adc,
    "BestMethod": bestmethods,
    "BestFitR2": bestr2,
}
df = pd.DataFrame(output)
df.to_excel('bestmethods.xlsx')

print("Summary")
print(f"Total Number of Countries: {len(adc)}")
print(f"Most Common Best Method: {mode(bestmethods)}")
