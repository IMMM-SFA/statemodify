---
title: 'statemodify: a Python framework to facilitate accessible exploratory modeling for discovering drought vulnerabilities'
tags:
  - Python
  - StateMod
authors:
  - name: Rohini S. Gupta
    orcid: 0000-0003-3538-0675
    affiliation: 1
  - name: Chris R. Vernon
    orcid: 0000-0002-3406-6214
    affiliation: 2
  - name: Travis Thurber
    orcid: 0000-0002-4370-9971
    affiliation: 2
  - name: David F. Gold
    orcid: 0000-0002-0854-1819
    affiliation: 1  
  - name: Zachary M. Hirsch
    orcid: 0009-0007-5356-5354
    affiliation: 3      
  - name: Antonia Hadjimichael
    orcid: 0000-0001-7330-6834
    affiliation: 4 
  - name: Patrick M. Reed
    orcid: 0000-0002-7963-6102
    affiliation: 1
        
        
affiliations:
 - name: Department of Civil and Environmental Engineering, Cornell University, 527 College Ave, Hollister Hall, Ithaca, NY, 14853, USA
   index: 1
 - name: Pacific Northwest National Laboratory, Richland, WA., USA
   index: 2
 - name: Department of Environmental Sciences and Engineering, Gillings School of Global Public Health,University of North Carolina at Chapel Hill, Chapel Hill, NC., USA
   index: 3   
 - name: Department of Geosciences, Penn State University, State College, PA., USA
   index: 4   
date: 15 December 2023
bibliography: paper.bib
---

# Summary
The Colorado River Basin (CRB) is experiencing an unprecedented water shortage crisis brought upon by a combination of factors arising from interactions across the region's coupled human and natural systems. Allocation of water to the seven states that rely on the Colorado River was settled in the Colorado River Compact of 1922 during a period now known to be characterized by atypically high flows [@christensen2004effects]. Since then, aridification due to anthropogenic-driven warming has steadily reduced the overall water supply available in the basin, with a 10% decrease in the river's flow occurring over just the past two decades [@bass2023aridification]. The river is further strained by increasing demands associated with a growing population and diverse multi-sectoral demands. Navigating these challenges also requires accounting for the complex prior appropriation water rights system governing water allocation across the region's diverse users. 

The state of Colorado's West Slope basins are a critical component of the Colorado River System and reflect the broader challenges faced by the entire basin. The six West Slope basins – the Upper Colorado, Yampa, White, San Juan, Dolores, and Gunnison basins – comprise the headwaters of the Colorado River and contribute over 60% of the inflow to Lake Powell in an average year [@salehabadi2020future]. The West Slope basins represent an essential part of the State of Colorado's economy, supporting a multibillion-dollar tourism industry, providing water for roughly 800,000 acres of irrigated farmland, and sending drinking water across the continental divide to major metropolitan areas in eastern Colorado [@CWCB-2023]. Uncertainty stemming from climate change and institutional response plays a dominant role in evaluations of future deliveries to Lake Powell and characterization of the basin users' vulnerabilities [@hadjimichael2020advancing; @hadjimichael2020defining; @salehabadi2020future]. Recent studies estimate that changes in temperature and precipitation may result in streamflows that are between 5% and 80% lower by the end of the 21st century when compared to the historical record [@kopytkovskiy2015climate; @milly2020colorado; @miller2021will]. Institutional responses to changes in flow, such as changes to reservoir operations and water rights structures, are difficult to predict and model using traditional probabilistic methods [@hadjimichael2020advancing; @hadjimichael2020defining]. This difficulty in accurately characterizing key system inputs with known probability distributions is often described as conditions of "deep uncertainty" [@lempert2002new; @kwakkel2016coping]. 

To account for the deeply uncertain future conditions in the West Slope basins, approaches are needed that can help facilitate an understanding of vulnerabilities across many plausible future scenarios [@lempert2002new; @walker2003dealing; @marchau2019decision]. Exploratory modeling is one such approach that uses computational experiments to understand a range of possible model behaviors [@bankes1993exploratory]. In the West Slope basins, exploratory modeling can be done with StateMod, a highly resolved, open source, regional water allocation model developed and maintained jointly by the Colorado Water Conservation Board (CWCB) and the Colorado Division of Water Resources (DWR) that is currently used to support water use assessments for the State of Colorado. The input files of StateMod can be manipulated to develop hypothetical scenarios to assess how changes in hydrology, water rights, or infrastructure impact regional water shortages, streamflow, or reservoir levels. 

StateMod is written in Fortran and conducting large ensemble exploratory modeling with it, on high-performance computing (HPC) resources, requires familiarity with Linux. Due to the model's complexity, there are also nontrivial computational challenges in comprehensively sampling the model's input space and managing the outputs of interest, especially for large ensembles. These challenges limit its use among researchers and broader operational users. Thus, we develop `statemodify`, a Python-based package and framework that allows users to easily interact with StateMod using Python exclusively. The user can implement `statemodify` functions to manipulate StateMod's input files to develop alternative demand, hydrology, infrastructure, and institutional scenarios for Colorado's West Slope basins and run these scenarios through StateMod. We also create methods to compress and extract model output into easily readable data frames and provide guidance on analysis and visualization of output in a series of Jupyter notebooks that step through the functionality of the package. 


# Design and Functionality

Figure 1 illustrates a typical `statemodify` workflow along with the corresponding functions that can be utilized in each step. Documentation of all functions (including helper functions that are not described in the figure) can be found in the `statemodify` [API](https://immm-sfa.github.io/statemodify/reference/api.html#input-modification). 


![`statemodify` workflow](JOSS.png)

# Statement of Need

Sustainable management of Colorado's West Slope basins is necessary to support inflow into the Colorado River and, by extension, the 40 million people that depend on it for water, hydropower, agriculture, and recreation [@Flavelle2023]. Because it is unknown how the future will manifest in the West Slope, exploratory modeling with StateMod is a valuable approach to comprehensively identify the most important drivers of change and vulnerabilities to different stakeholders. Sustainable management of the region will also ultimately require combining expert knowledge across diverse groups, ranging from federal and state institutions who are prescribing larger policy and conservation efforts down to the practical knowledge acquired from individual stakeholders, many of whom have livelihoods that have been supported by the river for many generations. In an effort to better maintain StateMod and expand their user base, CWCB and DWR have developed the CDSS Open Source Initiative ([OpenCDSS](https://opencdss.state.co.us/opencdss/)), which provides Java-based [TSTool](https://opencdss.state.co.us/opencdss/tstool/) and [StateDMI](https://opencdss.state.co.us/opencdss/statedmi/) software to create and modify StateMod input files. The `statemodify` package seeks to complement and expand this toolset to accommodate large ensemble exploratory modeling and a Linux-based workflow and to provide additional options to develop more targeted file adjustments and richer streamflow scenarios. The use of Python for all `statemodify` methods as well as the ease of interacting through Jupyter notebooks can further broaden the user base that can interact with StateMod and serve as a sandbox environment for quickly testing hypotheses that a user might have without the user needing to invest large amounts of time to learn how to use StateMod and develop a workflow. Examples of such questions could be: *What happens to user shortages in the Gunnison if evaporation rates were to change over the Blue Mesa Reservoir? If user X gains a more senior water right, does this have a bigger impact on reducing their shortages than shifts to a wetter hydroclimate?* 

Though `statemodify` focuses on Colorado's West Slope basins, this case study is representative of a broader group of institutionally complex basins that are experiencing extreme conditions due to their own regionally-specific deep uncertainties. Discovering vulnerabilities of users in these regions will likely require similar finely resolved models along with extensive computational resources (i.e. see CalSim [@draper2004calsim], WEAP21 [@yates2005weap21], MODSIM [@labadie2006modsim], CALFEWS [@zeff2021california]). If a user has access to a given model's input and output files, many of the methods introduced in `statemodify` could be adapted for adjusting input files and compressing and visualizing the output files. It is important to note that not every user has access to the computational resources required to do exploratory modeling. At this point in time, those users are unable to participate in this type of modeling effort. To help overcome this barrier, we demonstrate `statemodify` and the associated notebooks in containers hosted by [MSD-LIVE](https://msdlive.org/) that allow any user to conduct a small-scale analysis. Users can also use the associated Dockerfile to download and compile StateMod and run the same Jupyter notebooks on their own personal computers or HPC resources. Work is currently in progress to connect `statemodify` with cloud platforms, such as Amazon Web Services, Microsoft Azure, and Google Cloud, which provide more accessible tiers of computing to students and researchers who do not have access to HPC. Overall, the `statemodify` framework will not only broaden the user base that can interact with StateMod, but also can serve as a guide on how to make exploratory modeling accessible to diverse groups whose inclusion can lead to more robust basin management.        

# Acknowledgements
This research was supported by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program.

# References
