Useful Links
--------------------------------

Planning and Management Models
+++++++++++++++++++++++++++++++

The Colorado Water Conservation Board (CWCB) and the Division of Water Resources (DWR) have jointly developed Colorado's Decision Support System (CDSS), a collection of databases, data viewing and management tools, and models to support water resources planning in Colorado's major water basins (Malers et al., 2001). The CDSS is made up of a central database with water resources data (HydroBase), a public website where the data can be accessed, a Geographic Information System (GIS) for viewing and analyzing the data, and a consumptive use model (StateCU) that estimates consumptive use by each irrigation unit in a basin. The outputs from StateCU are then input to the State of Colorado's Stream Simulation Model (StateMod), a generic network-based water system model for water accounting and allocation, and the final component of CDSS. StateMod was developed to support comprehensive assessments of water demand, allocation, and use, as well as reservoir operations. It represents all of the major sub-basins within the state of Colorado (i.e., Parsons & Bennett, 2006; White, Yampa, Upper Colorado, Gunnison, Dolores, San Juan, and San Miguel CWCB, 2012). StateMod replicates each basin's unique application and enforcement of the prior appropriation doctrine and accounts for all of the consumptive use within the basins. To do so, it relies on the detailed historic demand and operation records contained in HydroBase that include individual water right information for all consumptive use, data on water structures (wells, ditches, reservoirs, and tunnels), and streamflow data. Further, StateMod uses irrigation consumptive use data output from StateCU, which calculates water consumption based on soil moisture, crop type, irrigated acreage, and conveyance and application efficiencies for each individual irrigation unit in the region.

**statemodify** complements CDSS tools such as TSTools and StateDMI (available on the `OpenCDSS <https://opencdss.state.co.us/opencdss/>` website) by focusing on providing tools that accommodate large ensemble exploratory modeling and a Linux-based workflow and to provide additional options to develop more targeted file adjustments and richer streamflow scenarios.



Exploratory Modeling
+++++++++++++++++++++++++++++++

Many of the methods and techniques displayed in the **statemodify** package have been debuted in peer-reviewed literature. Please see the following publications for more information:

Hadjimichael, A., Quinn, J., Wilson, E., Reed, P., Basdekas, L., Yates, D., & Garrison, M. (2020). Defining robustness, vulnerabilities, and consequential scenarios for diverse stakeholder interests in institutionally complex river basins. Earth's Future, 8(7), e2020EF001503.

Quinn, J. D., Hadjimichael, A., Reed, P. M., & Steinschneider, S. (2020). Can exploratory modeling of water scarcity vulnerabilities and robustness be scenario neutral?. Earth's Future, 8(11), e2020EF001650.

Hadjimichael, A., Quinn, J., & Reed, P. (2020). Advancing diagnostic model evaluation to better understand water shortage mechanisms in institutionally complex river basins. Water Resources Research, 56(10), e2020WR028079.



References
---------------------------
Malers, S. A., Bennett, R. R., & Nutting-Lane, C. (2001). Colorado's decision support systems: Data-centered water resources planning and administration. In Watershed Management and Operations Management 2000 (pp. 1-9).

Parsons, R., & Bennett, R. (2006). Reservoir operations management using a water resources model. In Operating Reservoirs in Changing Conditions (pp. 304-311).
