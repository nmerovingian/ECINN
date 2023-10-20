# ECINN

A  repository for **Discovering Electrochemistry with an Electrochemistry-Informed Neural Network (ECINN)**.  

# Requirements
Python 3.8 and above is required to rum the progrmas. The neural network was developed and tested with TensorFlow 2.4 and should be compatble with higher versions. Packagaes like Pandas, Numpy and Matplotlib are required. To visualize the structure of ECINN, pydot is required. 

# Folders
There are three folders for three cases, where ECINN was used to analyze real experimental data. They are:

* ECINN-BV for Fe Ion on GCE: An ECINN-BV is a neural network embedding both Butler-Volmer (BV) equation along with the diffusion equation, for multi-task discovery of electrochemical rate constants, transfer coefficients and diffusion coefficients. Since Fe<sup>2+</sup>/Fe<sup>3+</sup> redox reaction on glassy carbon electrode (GCE) is irreversible, only the cathodic term of the BV equation was adopted to extract $\alpha$. 

* ECINN-BV for Fe Ion on PtE: Fe<sup>2+</sup>/Fe<sup>3+</sup> redox reaction on Platinum electrode (PtE) was quasi-reversible, ECINN-BV adopted the full BV formalism to find electrochemical rate constants, $\alpha$, $\beta$. The diffusion equation was also adopted to give estimates of diffusion coefficients. 
* ECINN-Nernst for RuHex Ion on GCE: ECINN-Nernst was a multi-task neural network embedding diffusion and Nernst equation to obtain formal potential and diffusion coeffcients. The electrochemical reaction of Ru(NH<sub>3</sub>)<sub>6</sub><sup>2+</sup>/Ru(NH<sub>3</sub>)<sub>6</sub><sup>3+</sup>on GCE is fully reversible. 
 

## Experimental data and weights 
The original experimental data can be located in each .xlsx sheet in each folder. The weights of neural networks to fully reproduce the cases reported in paper can be found in the weights folder of each case folder. 

# Issue Reports
Please report any issues/bugs of the code in the discussion forum of the repository or contact the corresponding author of the paper




