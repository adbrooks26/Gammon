# Gammon -- Gamma-spec data generation tool

``` terminal

                  =@@@@@@@@@@@@@:                   
            .@@%%@%@@%%@%%%%%%  . @                 
         @@@%%@#%@+%*@+%+@+% .::...@                
       @@@%%%%%%*+***%+@*@+ ::..::::@               
     %@@%%@%%%%%%%%#*%**+* ..::.::::.:              
    @@@%@%@%%%%%%%%@%@%@%+....:::::--@              
   @@@@@%@%%%@#%%%%@%@#@% .::-:----=-: *+           
   @@@@%%%%%%%%%%%%@#%###::::=-=-==++=@@@@@         
   @@@@@%@%%%%@%%@%%%#@%%:::-==+++++*+---:. :       
   @@@@@@@*%@%@@@%%%%%%%:--==++++**##+==--.. :::.   
 +@%@@@@@@@%#@%%%@%%%@%%--=+***=####*===--:... :::  
 -@@@@@@@#@@@@@:.  %%%@%%=**####%%#*=-.:......:.. : 
 .:@@@@@@@@@@. :.  . :. #=**%%%@*#+:@%@%@  ::.... - 
  .-=-@@@@%@%. :. . . ... ##%%***:%#@@@@@@@@... @:. 
    .%@@@=##@@@@  .: :-::.= *#+@%=@@ @@@@#@@@@@:.   
      .::**@@@@@-.. .%:-..@=@%.@@:@@ @@@@@*::..     
          -:    =@@@ *@@@@@-                        
                   .                                
```

## Implementation

The datagen.py file is the generation function used to make gamma spec. It has the following inputs:

``` terminal
Parameters
----------

photopeaklist       : array_like(float or int) or float or int
                        Array of all photopeaks from desired source (in keV)

photopeakscale      : array_like(float or int) or float or int
                        Array of corresponding heights/scales of each photopeak  

comptonstrength     : float
                        Strength of compton continuum/continuua in gamma spectrum

xraylist            : array_like(float or int) or float or int
                        Array of all x-rays from desired source (in keV)

xrayscale           : array_like(float or int) or float or int
                        Array of corresponding heights/scales of each x-ray 

resolution_var      : array_like(float or int)
                        Resolution curve coefficients in log space from fitted resolution curve -- [p0,p1]

calibration_var     : array_like(float or int)
                        Calibration curve coefficients to convert from E to ch -- [p0,p1,p2]

noiseStr            : float 
                        Strength of gaussian noise applied to histogrammed data

backscatterOn       : bool
                        Flag to include or exclude the backscatter peak (Default == False)

bsStr               : int or float
                        Strength of backscatter peak(s) in spectrum

torchconvert         : bool
                        Flag to output data in a pytorch format
```

The outputs are then

``` terminal
Outputs
-------
xdata               : array_like
                        Channels (bin edges) of binned histogram data

counts              : array_like
                        Counts in each histogram bin
```

This function is mostly physics driven with it generating Gaussian peaks corresponding to the energies provided in the photopeak list, generates compton continua for each photopeak and then adds in xray peaks. A Gaussian noise is added to the final spectra which is rebinned to resemble the output from an MCA. The array can be converted into a torch array for inputting into pytorch data classes.

See the example.py file for how to implement the datagen class in pytorch machine learning algorithms. This file shows how the dataclass is setup and how labelled data for training are generated. This can be modified as required for different sources when needed.
