import numpy as np
from scipy import interpolate as interpl


def GammaSpecConstruct(photopeaklist,photopeakscale,comptonstrength,xraylist,xrayscale,resolution_var,calbration_var,noiseStr,backscatterOn=False,bsStr=0,torchconvert=False):
    """
    Function that constructs a histogrammed gamma spectrum (including photopeaks, compton, backscatter and escape peaks) which can then be used to generate mass training data for a ML algorithm

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

    Outputs
    -------
    xdata               : array_like
                            Channels (bin edges) of binned histogram data

    counts              : array_like
                            Counts in each histogram bin
    """



    def fwhm_from_rescoeff(centroidval,res_p0val,res_p1val):
        """
        Function that converts resolution log space plot coefficients into a fwhm

        Parameters
        ----------
        centroidval         : array_like or float or int
                                Energy to calculate the fwhm at

        res_p0val           : float or int
                                Intercept value of resolution plot in log log space
        
        res_p1val           : float or int
                                Gradient value of resolution plot in log log space

        Outputs
        -------
        array_like or float or int
                Fwhm at specified energy or energies
        """
        return np.exp((res_p1val*np.log(centroidval) + res_p0val))*centroidval 
        # log(Resolution) = log(FWHM(E)/E) = p1*log(E) + p0 solved for FWHM


    def photopeak(energylist,centroid,scale,res_p0,res_p1):
        """
        function that produces a number of Gaussian photopeaks

        Parameters
        ----------
        energylist      : array_like
                            Full energy array to calculate over (in keV)

        centroid        : array_like(float or int) or float or int
                            Photopeak centroids (in keV)

        scale           : array_like(float or int) or float or int
                            Scales of each photopeak

        res_p0          : float or int
                            Intercept value of resolution plot in log log space

        res_p1val       : float or int
                            Gradient value of resolution plot in log log space

                            
        Outputs
        -------
        array_like
            Clean gamma spectrum of desired photopeaks

        """

        spectrum = np.zeros_like(energylist) # create a spectrum np.array that will be used to store the counts at each energy
        peakfwhm = fwhm_from_rescoeff(centroid,res_p0,res_p1)   # calculate the peak FWHM 

        if (isinstance(centroid,list) or isinstance(centroid,np.ndarray)) and (isinstance(scale,list) or isinstance(scale,np.ndarray)): 
            # do an instance check to handle different types of energy input
            for i,eng in enumerate(centroid):   # loops through each energy provided in the list
                if eng < 2*511.:    # check if energy is above the pair production threshold
                    spectrum += scale[i]*np.exp(-(energylist-eng)**2 / (2*(peakfwhm[i]/2.35)**2))   # if not just save Gaussian to spectrum
                else:   # if it is above the pair production threshold put the Gaussian photopeak but also include the two escape peaks
                    single_esc_str = 0.001    # hard coded strength factors relative to the main photopeak -- adjust these as necessary, normally not visible so set to near zero
                    double_esc_str = 0.0001   # in reality these are cross section dependent
                    spectrum += (scale[i]*np.exp(-(energylist-eng)**2 / (2*(peakfwhm[i]/2.35)**2))
                                    + single_esc_str*scale[i]*np.exp(-(energylist-(eng - 511.))**2 / (2*(fwhm_from_rescoeff(eng-511.,res_p0,res_p1)/2.35)**2))
                                    + double_esc_str*scale[i]*np.exp(-(energylist-(eng - 2*511.))**2 / (2*(fwhm_from_rescoeff(eng-2*511.,res_p0,res_p1)/2.35)**2)))
            return spectrum # return full spectrum counts
        
        elif (isinstance(centroid,float) or isinstance(centroid,int)) and (isinstance(scale,float) or isinstance(scale,int)):
            # this does the same as above but handles the energy list if it is an integer of float rather than a numpy array
            if centroid < 2*511.:    
                spectrum += scale*np.exp(-(energylist-centroid)**2 / (2*(peakfwhm/2.35)**2))
            else:
                single_esc_str = 0.2    # again, if different ratios of escape peaks from pair production are required change here
                double_esc_str = 0.05
                spectrum += (scale*np.exp(-(energylist-centroid)**2 / (2*(peakfwhm/2.35)**2))
                                + single_esc_str*scale*np.exp(-(energylist-(centroid - 511.))**2 / (2*(((res_p1*(centroid-511.) + res_p0)/(centroid-511.))/2.35)**2))
                                + double_esc_str*scale*np.exp(-(energylist-(centroid - 2*511.))**2 / (2*(((res_p1*(centroid-2*511.) + res_p0)/(centroid-2*511.))/2.35)**2)))
                
            return spectrum
            
        else:
            return TypeError("centroid and scale are not type list/np.ndarray or float/int") # if input is incorrect formatting return a TypeError
        

    def compton(totenglist,photopeaks,comptonStr,res_p0,res_p1):
        """
        function that produces a number of Compton continuum and backscatter peak

        Parameters
        ----------
        totenglist      : array_like
                            Full energy array to calculate over (in keV)

        photopeaks      : array_like(float or int) or float or int
                            Photopeak centroids (in keV)

        comptonStr      : float or int
                            Scale of Compton continuum

        res_p0          : float or int
                            Intercept value of resolution plot in log log space

        res_p1val       : float or int
                            Gradient value of resolution plot in log log space

                            
        Outputs
        -------
        array_like
            Clean gamma spectrum of desired Compton continuua and backscatter peaks

        """

        compSpectrum = np.zeros_like(totenglist) # create an empty list to save counts to
        
        if isinstance(photopeaks,list) or isinstance(photopeaks,np.ndarray): # instance handling check for the photopeak energy again
            for i,photopeakE in enumerate(photopeaks):
                thetalist = np.linspace(0,np.pi,num=10000,endpoint=True)    # creates a high granularity scattering angle array to evaluate compton scattering
                re = 2.8179403205#e-15  # this is the classical radius of the electron, in fm 
                a = photopeakE/511. # this is the ratio of the photopeak energy to the electron mass
                problist_angle = 53 * re**2 * (1/(1+a*(1-np.cos(thetalist))))**2 * ((1+np.cos(thetalist)**2)/2) * (1 + (a*(1-np.cos(thetalist)))**2/((1+np.cos(thetalist)**2)*(1 + a*(1-np.cos(thetalist)))))#i.e. klein-nishina formula to look at probability of emission at given theta angle. Assuming a Z = 53 for NaI
                # this calculates the angular distribution from compton scattering -- i.e. the Klein-Nishina formula
                problist_eng = problist_angle*2*np.pi*(1 + a*(1-np.cos(thetalist)))**2 / (photopeakE*a) # converts the Klein-Nishina formula to terms of energy rather than angles
                ephoton = photopeakE/(1+a*(1-np.cos(thetalist)))    # obtains the photon energy from compton scattering
                eelectron = photopeakE - ephoton    # calculates the electron energy for a photon scattering under compton 

                #interpolate onto large grid so can be added back together... 
                preinterp = interpl.splrep(eelectron,problist_eng)
                interpedCompton = interpl.splev(totenglist,preinterp,ext=1)

                if backscatterOn == True:   # checks to see if the backscatter flag is on to then include the backscatter peak in the spectrum
                    backscatterElist = ephoton[int(10000/2)::]  # calculable from compton formula -- backscattering is just compton scattered event that comes into the detector
                    # this just takes the photon energy from a compton scatter and then says only events with angles > 90 will contribute to backscatter
                    backscatterEproblist = problist_eng[int(10000/2)::]
                    # this is just the corresponding probabilities of a photon of this energy being scattered from compton scattering
                    sort_idx = np.argsort(backscatterElist) #sorting to solve issue with interpolation input 
                    # -- sorts the input x values so that x is only increase (a requirement of the scipy.optimize function)
                    sorted_bsElist = backscatterElist[sort_idx] # maps the lists after sorting
                    sorted_bsEproblist = backscatterEproblist[sort_idx]

                    SpBSProbE = interpl.splrep(sorted_bsElist,sorted_bsEproblist)   # interpolates the backscatter energy
                    interpolatedBS = interpl.splev(totenglist,SpBSProbE,ext=1)

                    compSpectrum += comptonStr*interpedCompton + bsStr*interpolatedBS # adds the counts to the spectrum

                else:    
                    compSpectrum += comptonStr*interpedCompton  # if the backscatter flag is off just return the compton

            conv = np.zeros_like(compSpectrum)  
            # The generated compton is a perfect compton however our detector has a resolution so we now smear the compton by the detector energy dependent resolution.
            window_factor = 5  # truncate Gaussian at +-5 sigma for convolution to increase computation time

            for i, eng in enumerate(totenglist):    # loops over the energy array
                sig = fwhm_from_rescoeff(eng,res_p0,res_p1)/2.35    # calculates the sigma for the gaussian
                
                # Select points within ±5 sigma
                mask = (totenglist >= eng - window_factor*sig) & (totenglist <= eng + window_factor*sig)
                E_local = totenglist[mask]  # creates a mask and applies the mask the energy and spectrum
                S_local = compSpectrum[mask]
                
                # Gaussian weights
                weights = np.exp(-(E_local - eng)**2 / (2*sig**2))  # calculates the Gaussian weights
                weights /= weights.sum()  # normalises the Gaussian
                
                conv[i] = np.sum(weights * S_local) # performs the convolution

            return conv # returns the convoluted compton spectrum

        elif isinstance(photopeaks,float) or isinstance(photopeaks,int):    # handles the same compton process but for integer or float photopeak energy input
            thetalist = np.linspace(0,np.pi,num=10000,endpoint=True)
            re = 2.8179403205#e-15
            a = photopeaks/511.
            problist_angle = 53 * re**2 * (1/(1+a*(1-np.cos(thetalist))))**2 *((1+np.cos(thetalist)**2)/2) * (1 + (a*(1-np.cos(thetalist)))**2/((1+np.cos(thetalist)**2)*(1 + a*(1-np.cos(thetalist)))))#i.e. klein-nishina formula to look at probability of emission at given theta angle. Assuming a Z = 53 for NaI
            problist_eng = problist_angle*2*np.pi*(1 + a*(1-np.cos(thetalist)))**2 / (photopeaks*a)
            ephoton = photopeaks/(1+a*(1-np.cos(thetalist)))
            eelectron = photopeaks - ephoton

            preinterp = interpl.splrep(eelectron,problist_eng)
            interpedCompton = interpl.splev(totenglist,preinterp,ext=1)

            if backscatterOn == True:
                backscatterElist = ephoton[int(10000/2)::]
                backscatterEproblist = problist_eng[int(10000/2)::]
                sort_idx = np.argsort(backscatterElist)
                sorted_bsElist = backscatterElist[sort_idx]
                sorted_bsEproblist = backscatterEproblist[sort_idx]

                SpBSProbE = interpl.splrep(sorted_bsElist,sorted_bsEproblist)
                interpolatedBS = interpl.splev(totenglist,SpBSProbE,ext=1)

                compSpectrum += comptonStr*interpedCompton + bsStr*interpolatedBS

            else:    
                compSpectrum += comptonStr*interpedCompton
            
            conv = np.zeros_like(compSpectrum)
            window_factor = 5 

            for i, eng in enumerate(totenglist):
                sig = fwhm_from_rescoeff(eng,res_p0,res_p1)/2.35

                mask = (totenglist >= eng - window_factor*sig) & (totenglist <= eng + window_factor*sig)
                E_local = totenglist[mask]
                S_local = compSpectrum[mask]
                
                weights = np.exp(-(E_local - eng)**2 / (2*sig**2))
                weights /= weights.sum() 
                
                conv[i] = np.sum(weights * S_local)

            return conv


        else:
            return TypeError("centroid is not of type list/np.ndarray or float/int") # if input is incorrect return a TypeError


    def digitise(totenglist,spectrum,calibrationcoeff,maxch,noiselevel):
        """
        function that produces a turns clean gamma spectra into histogrammed data with noise to replicate experimental data taken

        Parameters
        ----------
        totenglist          : array_like
                                Full energy array to calculate over (in keV)

        spectrum            : array_like
                                Spectrum of photopeaks and Compton 

        calibrationcoeff    : array_like(float or int)
                                Coefficients of calibration curve to convert from energy to channel

        maxch               : int
                                Maximum channel number of histogram

        noiselevel          : int
                                Strength of generated noise

                            
        Outputs
        -------
        digitisedch         : array_like
                            Channels (bin edges) of binned histogram data

        digitiseddata       : array_like
                                Counts in each histogram bin with noise applied

        """
        if isinstance(calibrationcoeff,list) or isinstance(calibrationcoeff,np.ndarray): # checks to see if calibration coeff are a list or array -- otherwise returns TypeError
            chlist = calibrationcoeff[2]*totenglist**2 + calibrationcoeff[1]*totenglist + calibrationcoeff[0]   # performs a reverse calibration (energy -> channel)
            
            # convert back into a histogram to look like/act like normal experimental data:

            rounded_spectrum = np.array([round(i) for i in spectrum]) #integerise spectrum to resemble counting data

            bins_list = np.arange(0,maxch,1)    # create a bin list 0->2047

            #interpolate back out of highly binned data to prevent weird artefacting that can happen when binning high granularity data
            splineinterpeddata = interpl.splrep(chlist,rounded_spectrum)
            uninterpolateddata = interpl.splev(bins_list,splineinterpeddata,ext=1)
            # then pass the data through np.histogram to have it in a histogrammed format... 
            digitiseddata,digitisedch = np.histogram(bins_list,weights=uninterpolateddata,bins=maxch,range=(0,maxch-1))
            # finally return the channels and the data + some noise applied to the data -- currently applies 3 sets of noise to remove any patterns -- this is controlled by the noiselevel variable
            # apply an absolute to the noisy spectrum to prevent any negative values from coming out... 
            return digitisedch,np.abs(digitiseddata + (noiselevel*np.random.normal(np.random.uniform(0,2),np.random.uniform(0,1),2048) + np.random.normal(np.random.uniform(0,1),np.random.uniform(0,2),2048) + np.random.normal(np.random.uniform(0,1),np.random.uniform(0,1),2048)))
        else:
            return TypeError("calibration coefficients must be a list or np.ndarray")


    def dtorchconverter(data):
        """
        Converter function to convert input 1d hist data into format suitable for pytorch to use
        
        Parameters
        ----------
        data          : array_like
                            1d hist data to be converted

        
        Outputs
        -------
        array_like
                    converted data
        """
        converteddata = [] # creates an empty array for data formatting
        for value in data:  # loops over every value in the input arr 
            converteddata.append([value])   # maps it to the following format [[],[],[]] rather than just [, , ,]
        return np.asarray(converteddata)    # this is for pytorch input requirements
        

    #################################

    energyarr = np.linspace(0.05,2000,num=10001,endpoint=True) #may want to do an arange later or be able to change the maximum energy value... 
    # creates an energy array defined from near zero to 2000keV, this can be adjusted as desired if higher energy gamma rays are needed 

    spectrumprofile = photopeak(energyarr,photopeaklist,photopeakscale,resolution_var[0],resolution_var[1]) # generates the photopeaks
    xraysprofile = photopeak(energyarr,xraylist,xrayscale,resolution_var[0],resolution_var[1])  # adds on x-rays through the same process as the photopeaks
    spectrumprofile += xraysprofile # combines these 

    comptonprofile = compton(energyarr,photopeaklist,comptonstrength,resolution_var[0],resolution_var[1])   # calculates the compton for the provided photopeaks
    spectrumprofile += comptonprofile   # adds to spectrum

    xdata,counts = digitise(energyarr,spectrumprofile,calbration_var,2048,noiseStr) # digitises the data to histogram format

    if torchconvert:    # checks for torch requirement to then apply formatting to data if True
        convertedcounts = dtorchconverter(counts)
        return xdata+0.5,convertedcounts    # returns the bins and data. bins are shifted by 0.5 to make the spectrum start at 0.0 rather than -0.5

    else:
        return xdata+0.5,counts

# Below is some generic testing code that just checks things and can also plot the spectrum to check it is all ok... 
# code testing to make sure everything works for torch inputs... 


# xarr,spect=GammaSpecConstruct(662,100*1.0,10.0,[0.75,-0.522],[14.06,1.028,0.0],5,backscatterOn=True,bsStr=0.5,torchconvert=True)
# print(spect, len(xarr))

### test code to check that everything plots ok

# import matplotlib.pyplot as plt
# energylist,spect=GammaSpecConstruct(662,100*1.0,10.0,33,10*0.01,[0.75,-0.522],[14.06,1.028,0.0],5,backscatterOn=True,bsStr=0.5)

# plt.bar(energylist[:-1], spect, width=np.diff(energylist))

# plt.show()


####

def ParseSpec(filename,torchconvert=True):
    """
    Parsing function to convert .Spe data from 2048 ch MCA into an array which can be formatted for pytorch use

    Parameters
    ----------
    filename          : Str
                            Name of .Spe data file

    torchconvert      : Bool
                            Flag for converting data to torch formatting

    Outputs
    -------
    array_like
            Formatted data or list of counts in histogram bins

    """

    def dtorchconverter(data):
        """
        Converter function to convert input 1d hist data into format suitable for pytorch to use
        
        Parameters
        ----------
        data          : array_like
                            1d hist data to be converted

        
        Outputs
        -------
        array_like
                    converted data
        """
        converteddata = [] # creates an empty array for data formatting
        for value in data:  # loops over every value in the input arr 
            converteddata.append([value])   # maps it to the following format [[],[],[]] rather than just [, , ,]
        return np.asarray(converteddata)    # this is for pytorch input requirements
    

    outarray = []

    with open(filename) as f:
        for index,line in enumerate(f):
            if index >= 12 and index <= 2059:
                outarray.append(int(line.strip()))
    if torchconvert:
        convertedarr = dtorchconverter(outarray)
        return convertedarr
    else:
        return outarray


# print(ParseSpec("Data/Cs137_100s.Spe"))