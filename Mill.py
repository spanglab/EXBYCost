import biosteam as bst
import numpy as np


class Mill(bst.Unit):
    """
    Mill with cost correlation based on moisture content and
    power consumption based on flow rate, moisture, and particle size.

    Power equations:
        Knife mill (moisture < 30%): P = 166.67 * ṁ * e^(0.01*M) / (20*r)
        Wet ball mill (moisture >= 30%): P = 33 * P_knife_at_30%_moisture (constant)

        where ṁ is in kg/s, M is moisture %, r is particle radius in cm
        Output P is in kW

    Cost correlation (based on power):
        If moisture < 0.3: Knife mill ($11,250 @ 166 kW, 2006)
        If moisture >= 0.3: Wet ball mill ($100,000 @ 33.5 kW, 2006)

    Parameters
    ----------
    ins : Stream
        Inlet stream to be milled.
    outs : Stream
        Outlet stream.
    particle_radius : float
        Target particle radius [cm].
    moisture_content : float, optional
        Override moisture content calculation (mass fraction, 0-1).
        If None, calculated from water content in feed.
    n : float, optional
        Scaling exponent for cost correlation. Default is 0.667.
    power_coefficient : float, optional
        Coefficient in power equation. Default is 166.67.

    """
    _N_ins = 1
    _N_outs = 1
    _units = {'Power': 'kW', 'Particle radius': 'cm', 'Flow rate': 'kg/s'}

    # Cost parameters dictionary (same pattern as SolidSolventExtractor)
    MILL_COST_PARAMS = {
        'Knife mill': {
            'basis': 'Power',
            'units': 'kW',
            'cost': 11250,
            'CE': 499.6,
            'n': 0.664,
            'BM': 1.5,
            'S': 166,
        },
        'Wet ball mill': {
            'basis': 'Power',
            'units': 'kW',
            'cost': 100000,
            'CE': 499.6,
            'n': 0.664,
            'BM': 1.5,
            'S': 33.5,
        },
    }

    def _init(self, particle_radius, moisture_content=None, n=0.667,
              power_coefficient=166.67):
        self.particle_radius = particle_radius  # cm
        self._moisture_content = moisture_content
        self.n = n
        self.power_coefficient = power_coefficient

    def _run(self):
        self.outs[0].copy_like(self.ins[0])

    @property
    def moisture_content(self):
        """Calculate or return moisture content (mass fraction, 0-1)."""
        if self._moisture_content is not None:
            return self._moisture_content
        feed = self.ins[0]
        if feed.F_mass == 0:
            return 0
        return feed.imass['Water'] / feed.F_mass

    @property
    def moisture_percent(self):
        """Moisture content as percentage (0-100)."""
        return self.moisture_content * 100

    @property
    def mill_type(self):
        """Return mill type based on moisture content."""
        return 'Knife mill' if self.moisture_content < 0.3 else 'Wet ball mill'

    def calculate_power_knife_mill(self, mass_flow_kg_s, moisture_percent, particle_radius_cm):
        """
        Calculate power consumption for knife mill (dry).

        P = 166.67 * ṁ * e^(0.01*M) / (20*r)

        Parameters
        ----------
        mass_flow_kg_s : float
            Mass flow rate [kg/s]
        moisture_percent : float
            Moisture content [%]
        particle_radius_cm : float
            Target particle radius [cm]

        Returns
        -------
        float
            Power consumption [kW]
        """
        P = (self.power_coefficient
             * mass_flow_kg_s
             * np.exp(0.01 * moisture_percent)
             / (20 * particle_radius_cm))
        return P

    def calculate_power_wet_ball_mill(self, mass_flow_kg_s, particle_radius_cm):
        """
        Calculate power consumption for wet ball mill.

        Power is constant at 33x the knife mill power at 30% moisture.
        P = 33 * 166.67 * ṁ * e^(0.01*30) / (20*r)

        Parameters
        ----------
        mass_flow_kg_s : float
            Mass flow rate [kg/s]
        particle_radius_cm : float
            Target particle radius [cm]

        Returns
        -------
        float
            Power consumption [kW]
        """
        # Knife mill power at 30% moisture
        P_knife_30 = (self.power_coefficient
                      * mass_flow_kg_s
                      * np.exp(0.01 * 30)
                      / (20 * particle_radius_cm))
        # Wet ball mill is 33x that
        P = 33 * P_knife_30
        return P

    def _design(self):
        feed = self.ins[0]
        m_dot_kg_hr = feed.F_mass  # kg/hr (BioSTEAM default)
        m_dot_kg_s = m_dot_kg_hr / 3600  # Convert to kg/s
        M = self.moisture_percent  # %
        r = self.particle_radius  # cm
        mc = self.moisture_content  # fraction

        # Calculate power based on mill type
        if mc < 0.3:
            power = self.calculate_power_knife_mill(m_dot_kg_s, M, r)
        else:
            power = self.calculate_power_wet_ball_mill(m_dot_kg_s, r)

        self.design_results['Flow rate'] = m_dot_kg_s
        self.design_results['Particle radius'] = r
        self.design_results['Power'] = power
        self.power_utility.consumption = power

    def _cost(self):
        """Calculate equipment costs based on mill type.

        Uses the same pattern as SolidSolventExtractor: a cost parameters
        dictionary with explicit CEPCI adjustment, separating
        baseline_purchase_costs from purchase_costs.
        """
        Design = self.design_results
        equipment = self.mill_type
        params = self.MILL_COST_PARAMS[equipment]

        # Register bare-module factor
        self.F_BM[equipment] = params['BM']

        # Get the design value for costing basis
        design_value = Design[params['basis']]
        design_value = max(design_value, 1e-6)  # Avoid zero division

        # Scale the reference cost: C = C_ref * (S / S_ref)^n
        S_ref = params['S']
        C_ref = params['cost']
        n = params['n']

        scaled_cost = C_ref * (design_value / S_ref) ** n

        # Adjust for CEPCI
        current_CE = bst.CE
        ref_CE = params['CE']
        adjusted_cost = scaled_cost * (current_CE / ref_CE)

        self.baseline_purchase_costs[equipment] = scaled_cost
        self.purchase_costs[equipment] = adjusted_cost