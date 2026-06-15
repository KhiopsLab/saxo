# SAXO cost as the sum of bi-clustering costs

See notations in [^fn1]. Instead of using $m_i$ for the number of data points for each time series $S_i$, we use $p_s$ for the time series $s$ to avoid confusion with other $m_{i,j}$ quantities realted to the number of data points in cells $(i,j)$.

**SAXO Total Cost:**

$$
\begin{align*}
C_{\text{SAXO}}(M)=&\log m +\log\binom{m + k_T-1}{k_T -1} + \sum_{t=1}^{k_T}\log m^t + k_T\log N + \sum_{t=1}^{k_T}\log B(N,k_C^t)\\
&+\sum_{t=1}^{k_T}\log\binom{m^t + k_C^tk_X^t-1}{k_C^tk_X^t-1}+\sum_{t=1}^{k_T}\sum_{i=1}^{k_C^t}\log\binom{m_i^t+n_i^t-1}{n_i^t-1}\\
&+ 2\log m!-\sum_{t=1}^{k_T}\sum_{i=1}^{k_C^t}\sum_{j=1}^{k_X^t}\log m_{i,j}^t!  +\sum_{t=1}^{k_T}\left(\sum_{i=1}^{k_C^t}\log m_i^t!+\sum_{j=1}^{k_X^t}\log m_j^t! -\sum_{s=1}^{N}\log p_s^t! \right)
\end{align*}
$$

**Bi-clustering Sub-Cost:**

$$
\begin{align*}
C_{\text{MODL}}(M^t)=&\log m^t + \log N^t + \log B(N^t, k_C^t)\\
&+ \log\binom{m^t + k_C^t k_X^t - 1}{k_C^t k_X^t - 1} + \sum_{i=1}^{k_C^t}\log\binom{m_i^t-n_i^t-1}{n_i^t-1}\\ 
&+ \log m^t! - \sum_{i=1}^{k_C^t}\sum_{j=1}^{k_X^t}\log m_{i,j}^t!+\left(\sum_{i=1}^{k_C^t}\log m_i^t! + \sum_{j=1}^{k_X^t}\log m_j^t! - \sum_{s=1}^{N} \log p_s^t !\right)
\end{align*}
$$

**Difference:**

Assuming no missing values, meaning all time series are populated in every time intervals, $\forall t, N^t = N$:

$$
C_{\text{SAXO}}(M) - \sum_{t=1}^{k_T}C_{\text{MODL}}(M^t)=\log m + \log\binom{m + k_T-1}{k_T -1} + 2\log m! - \sum_{t=1}^{k_T}\log m^t !
$$

Otherwise (with missing-values, or sub-sampling):

$$
\begin{align*}
C_{\text{SAXO}}(M) - \sum_{t=1}^{k_T}C_{\text{MODL}}(M^t)=&\log m + \log\binom{m + k_T-1}{k_T -1} + 2\log m! - \sum_{t=1}^{k_T}\log m^t!\\
&+k_T\log N - \sum_{t=1}^{k_T}\log N^t\\ 
&+\sum_{t=1}^{k_T}\log B(N, k_C^t) - \sum_{t=1}^{k_T}\log B(N^t, k_C^t)
\end{align*}
$$

I think the SAXO cost is not precise enough in the paper because it makes too many assumptions for the sake of simplicity. If precise enough, differences between $N$ terms and $N^t$ terms should cancel out anyway.

**Null Cost:**

$$
\begin{align*}
C_{\text{SAXO}}(M_0) =& \log m + 0 + \log m + \log N + 0\\
&+ 0 + \log \binom{m + N - 1}{N-1} \\
&+2\log m!  - \log m! + \left(\log m! + \log m! - \sum_{s=1}^{N}\log p_s! \right)\\
\end{align*}
$$

Thus:

$$
C_{\text{SAXO}}(M_0) = 2\log m + \log N + \log \binom{m + N - 1}{N-1} + 3 \log m! - \sum_{s=1}^{N}\log p_s!\\
$$

Note that $C_{\text{SAXO}}(M_0)=C_{\text{MODL}}(M_0)$ where $C_{\text{MODL}}(M_0)$ corresponds to the null cost of triclustering [^fn2].

Nonetheless, I don't think we can compute the null cost of SAXO with the null cost of the bi-clustering models:

$$
\begin{align*}
C_{\text{MODL}}(M_0^t) =& \log m^t + \log N^t + 0\\
&+ 0 + \log \binom{m^t + N^t - 1}{N^t-1} \\
&+\log m^t!  - \log m^t! + \left(\log m^t! + \log m^t! - \sum_{s=1}^{N}\log p^t_s! \right)\\
\end{align*}
$$

So:

$$
C_{\text{MODL}}(M_0^t) = \log m^t + \log N^t + \log \binom{m^t + N^t - 1}{N^t-1} + 2 \log m^t! - \sum_{s=1}^{N}\log p^t_s!\\
$$

Problem, what if all bi-clusters $M^t$ are trained from different datasets $\mathcal{D}_t \neq \mathcal{D}$ with $\mathcal{D}=\bigcup_{t=1}^{k_T}\mathcal{D}_t$ ?

Heuristic: assumes that $m^t=\frac{m}{k_T}$, $p_s^t=\frac{p_s}{k_T}$ and $N^t=N$ ?

---

[^fn1]: Alexis Bondu, Marc Boullé, and Antoine Cornuéjols. "Symbolic representation of time series: A hierarchical coclustering formalization." International Workshop on Advanced Analytics and Learning on Temporal Data. 2015.

[^fn2]: Alexis Bondu, Marc Boullé and Benoît Grossin. "SAXO: An optimized data-driven symbolic representation of time series". International joint conference on neural networks (IJCNN) IEEE. 2013. 
