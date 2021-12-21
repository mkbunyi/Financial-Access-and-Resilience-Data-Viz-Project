# Examining Financial Resilience and Inclusion using World Bank Global Findex Data

Resilience is a concept that the COVID-19 pandemic has highlighted. The pandemic surfaced vulnerabilities across all sectors, especially for lower income households who are the hardest hit by financial and employment shocks and who rely on financial safety nets to weather the uncertainty and instability of the situation. Empowering the vulnerable to achieve financial resilience is key to lifting them out of poverty and helping them stay out of it. If we can determine the most important environmental factors that build financial resilience, then we can come up with better targeted interventions to curb the financial insecurity experienced by a significant proportion of the people and help them achieve upward economic mobility.

## Data

In analyzing financial inclusion, I use data from the World Bank Global Financial Inclusion (Global Findex) Database 2014 and 2017 as well as World Bank’s estimates for income inequality (Gini index) from the World Development Indicators (for 2017 or the latest year available prior). The Global Findex survey asks individual respondents about financial resilience, which is framed as the ability of the respondent to gather emergency funds within the next month of an appropriate amount set by the researchers.

```markdown
Interacting with the data visualizations:
There are various ways you can interact with the data. Feel free to explore and generate your own insights!
- Click on the individual data points to highlight that country, income level, or region.
- Click on a legend option to highlight that category.
- Filter to isolate only the relevant data points.
- Hover over the data points to show a tooltip containing that country’s details.
- Note that highlighting blurs but keeps the rest of the data points in the background, while filtering subsets the data points you wish to view.
```

## Tracing the links of financial resilience

<iframe seamless frameborder="0" src="https://public.tableau.com/views/1FinancialResiliencevsFinancialAccess/Dashboard1?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

Is financial access indicative of financial resilience? The map above indicates that it may not necessarily be so. While individuals in most countries with high financial access do tend to possess high financial resilience, it is also possible to have high access but remain low in resilience (as is the case for South Africa). The converse also holds – Vietnam is low in financial access but high in resilience.

To contrast compositions more easily across categories, we could group the countries by financial resilience/access and income level.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/1FinancialResiliencevsFinancialAccess/Dashboard2?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

Many countries across all income levels possess both high financial access and resilience. Low-income countries do not necessarily have low levels of access nor resilience. For most high-income countries, high financial access is a given (except for Uruguay, which is in the middle of the road). While generally positive, the relationship between financial access and resilience does not appear to be definitive. If financial access does not automatically correspond with financial resilience, how about inequality?

<iframe seamless frameborder="0" src="https://public.tableau.com/views/2FinancialResiliencevsIncomeInequality/Dashboard1?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

The relationship between financial resilience and income inequality appears stronger. Low inequality does not coexist with low financial resilience. Neither does high inequality with high financial resilience. This aligns with the conceptual link between poverty and vulnerability – those with lower wealth tend to face higher vulnerability and are unable to accumulate enough resources to weather emergencies. While financial access may equip households with the capacity to access resources from the financial system, this must accompany actual wealth formation. Service provision may not be enough; the expansion of the middle class is also necessary to foster financial resilience.

## Financial inclusion indicators
Drilling down into a more micro-perspective, how does financial access vary across countries? What individual-level factors link to the level of financial resilience in a country?

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/access?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

Disparities are evident across the different income groups, with financial access tending to increase as we go to a higher income level. Nonetheless, the generally wider financial access from 2014 to 2017 shows the progress made across all income level groups.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/inclusion?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

Despite the significant gaps across income levels in terms of financial account ownership and digital payment uptake, financial resilience (i.e., coming up with emergency funds) and savings behavior are more broadly similar for the low- and middle-income level groups. The rise of digital finance is also evident in the across-the-board increase in the use of digital payments.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/inclusion_poorest?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

Financial inclusion is most crucial for the poorest citizens of a nation, as this allows them access to resources of the financial system which traditionally, they would not have. For the income poorest 40%, albeit expectedly lower absolute numbers compared to the population, the range and trend of financial inclusion is not markedly lower.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/borrow?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

For borrowing behavior, the pattern for 2014 is broadly similar across the income levels, with the exception of less borrowings for medical or business purposes for the upper middle income and high-income groups. However, 2017 registered a significant increase in borrowing for business purposes in high-income countries. On the other hand, individuals in low-income countries decreased their borrowing behavior across both medical and business purposes. This may warrant a closer look into the reason behind the decrease – whether it is an indication of less need or less availability.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/save?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

For high-income countries, the positive correlation between financial resilience and saving behavior is more pronounced, especially for parking these funds at a financial institution. Saving at financial institutions as well as saving for retirement also have more stark differences across income levels. Similar to borrowing for business, saving for business increased in high income countries yet decreased for low-income countries.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/3WBGlobalFindex/reason?:embed=yes&:display_count=yes&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

Across all income levels, those who remain unbanked are so due mainly to a lack of funds. This ties up with the earlier link found between income inequality and financial resilience. The cost of financial services is also a major deterrent for the unbanked. For upper middle income and high-income countries, having someone in the family with an existing account also turned out as an important consideration behind the choice not to participate in the formal financial system.

## Conclusion
Financial resilience and financial inclusion are multi-faceted issues. While financial access in its different forms has increased through time, it does not appear to be enough. Overcoming financial vulnerability is associated with both the capacity to tap into the financial system and the capital that one can put into or receive from the system. 

Differences in the financial indicators show that countries have their own unique circumstances, but there are also similarities which may allow for cooperation and support. Knowledge of this baseline information on the unbanked and the financially vulnerable would help policymakers assess priority areas requiring intervention.

The link found between saving and financial resilience illustrates that social safety net programs should aim towards improving not just the financial strength but also the financial behavior of households across different income levels. Having the ability to save is an important cushion against shocks such as disasters, medical emergencies, and employment instability. Micro-credit opportunities may likewise merit a closer look to help the entrepreneurial poor tap into additional funding sources that would allow them to earn income and become more self-sufficient. This may be especially important in lower income countries where there is a low tendency for both borrowing and saving for business.







### Data Sources
[World Bank Global Findex Database 2014 and 2017](https://globalfindex.worldbank.org/)
[World Bank World Development Indicators](https://datatopics.worldbank.org/world-development-indicators/)

