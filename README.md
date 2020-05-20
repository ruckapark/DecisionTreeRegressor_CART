# Machine Learning CART Regression Tree

## Use code

I have included the readfile.py code which encapsulates the method, I am currently working on a more general method to adapt this code to any similar format csv file.

Below is an outline of the steps followed in the code (**readfile.py**)
#### Initialise
* set file name
* set plotting styles
#### Wrangling with Read data file with readfile format
* determines the type of data file (csv,xls etc.) and uses the appropriate pd.read functions
* returns the data frame of data
#### Data Cleaning
* Ensure index of df is the datetime object with D/M/Y H/M/S format
* Read in the temperature data file for the prediction and ensure same index format
* Merge the two dataframe so that temp becomes a column of the common df
* drop any na values (not useful for the learning methods to produce synthetic data)
* Create a weekday column (useful for this model and it is a dependant variable)
* Do the same for the hours and minutes in the day - using a lambda function make the hm into a continuous numeric scale so the ML can interpret this data more easily
* input if necessary the True False column for a base heating period (factories often have in the winter a higher minimum energy consumption to keep all pipes unfrozen in the night etc.)
* From a list of bank holidays - change the day of the week to sunday regardless of the day as no one will be in the office
#### Visualisation
I prefer to make a series of plots before the training of the data to ensure I have correctly understood the data. This can give you some idea of predictor variables and if they have been misdefined. Sometimes it is not this simple!  
PLOTS:
* Energy Consumption over known period through time
* Mean Daily Consumption
* Mean Weekly
* scatter distribution of temp and power rating
#### Machine Learning
* New dataframe with the appropriate predictor variables (not target values)
* Account for all bank holidays in the same way
* Clean dataframe to be correctly ordered and match previous
* import decision tree regressor and adaboost methods (adaboost optional)
* Create X and Y dfs
* Use sklearn test _ train split for X_train test Y etc.
* Use elbow method running through iteration of depths of regressor tree.
* Run model boosted with selected depths
#### Final Plots and Validation
* plot full year
* coherence plots - do low temps correspond to high power in all cases - if not why not?
* Validation calculations of the total energy consumption (area under curve) compared with the billing accounted for

*Code files to come in general form*:  
* format_data.py
* machine_learning.py