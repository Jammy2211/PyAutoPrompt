I am going to add weak gravitational lens analysis to autolens, whcih is going to be a series of 5 or so 
claude prompts to get us there.

First, we need to document the weak lensing code aleady in autolens, and make sure we know what it does. The good news is, 
the ability to generate a shear from the source code is already possible. All we need to do is use the 
shear_yx_2d_via_hessian_from in the module @PyAutoGalaxy/autogalaxy/operate/lens_calc.py. This returns the shear field which can then be the data
in a simulator.

This returns a shear field defined in @PyAutoGalaxy/autogalaxy/util/shear_field.py, which is how we store
shear fields of data in general.

The Isothermal profile, has the method shear_yx_2d_from in the profiles/mass/total/isothermal.py
module. This should have the same outptu as shear_yx_2d_via_hessian_from, but if there is no unit test afgainst that
please add one.

Of course the external shear for lens mass modeling in @PyAutoGalaxy/autogalaxy/profiles/mass/sheets/external_shear.py
will also help with thes defintions, make sure they are all consistent.

The main goal of this prompt to assess the existing shear calculatiojns in the source code, work out their API,
interface and other factors and write doctrings and documentaiton for them that will help the additon of weak lensing
to the code in general.
