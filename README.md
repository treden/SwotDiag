# Geostrophic Velocity Computation from Sea Surface Height (SSH) Data

## Overview

To compute current velocities from Sea Surface Height (SSH) data, we apply the geostrophic balance equations:

```math
u_g &= -\frac{g}{f}\frac{\partial SSH}{\partial y} \\
v_g &= \frac{g}{f}\frac{\partial SSH}{\partial x}
```

In most oceanographic applications, computing the first derivative at a central grid point involves differencing adjacent SSH in both the x and y directions. To enhance accuracy and correct for anisotropic artifacts, the stencil width of centered differences can be expanded from three points to five, seven, or nine points (Arbic et al., 2012).

Unlike previous gridded sea surface height products relying on combinations of along-track SSH profiles, SWOT provides unprecedented bi-dimensional SSH data on two large swathes of 50 km, with a 2 km pixel for the standard product. In alignment with the new two-dimensional paradigm offered by SWOT, we propose to calculate the slope by fitting a linear plane kernel to 2D SSH observations, as determined by:

\[
SSH_{fit} = a_1x + a_2y + a_3
\]

where \(a_1\) and \(a_2\) represent the across-track and along-track derivatives, respectively.

## Geostrophic Velocity Field

![Geostrophic Velocities](Figures/geostrophic_fit.png)
*Figure 1: Geostrophic velocities computed from noisy (a) and noiseless (b) SSH fields, using central point difference (left panels) and 2D SSH fit with kernel widths of 5, 9, and 13 points, respectively from left to right.*

By adjusting the kernel width, this method allows computation of derivatives that are more accurate and less noisy than the central grid point differences, on original (not denoised) and unsmoothed (250 m pixels) SWOT SSH data. Figure 1 shows the resulting geostrophic velocity field for central point difference and SSH fit methods, on the noisy original (a) and the denoised (b) SWOT SSHs.

In the same way, second derivatives obtained by fitting a quadratic surface instead of a linear plane (or fitting a surface on derivative fields) can be used for computing higher-order ocean diagnostics on SWOT swathes, such as relative vorticity or strain rates (see Figure \ref{fig:vorticity_fit} in the supplementary materials).

## Rotating Derivatives

To compute zonal and meridional derivatives, it is also necessary to rotate the derivatives by the angle \(\theta\), which is the SWOT pass inclination relative to the meridional axis, and defined as:

\[
\theta = \tan^{-1}\left(\frac{\delta x}{\delta y}\right)
\]

where \(\delta x\) and \(\delta y\) are the zonal and meridional distances between adjacent SWOT pixels. Then, the first derivatives can be expressed at each point as:

\[
\begin{align}
\frac{\partial SSH}{\partial x} &= a_1 \cos(\theta) - a_2 \sin(\theta) \\
\frac{\partial SSH}{\partial y} &= a_1 \sin(\theta) + a_2 \cos(\theta)
\end{align}
\]

## Mitigating Noise

We use the SSH fit method to mitigate the effects of measurement noise and non-balanced signals when deriving geostrophic velocities, as it acts as a low-pass filter on these velocities. To avoid the superimposition of low-pass filtering effects, we apply this method directly to the original (noisy) SWOT SSH field.

In the following analysis, we use a kernel width of 13 pixels, which corresponds to 26 km for the standard SWOT product and provides the best fit to the low-passed drifter velocities (see section \ref{impact_noiseless} of the supplementary materials). However, the sensitivity of the results to the kernel width is discussed in section \ref{discussion_kernel_length}.
