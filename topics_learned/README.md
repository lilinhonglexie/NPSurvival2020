# Topics Learned

Topics learned by our proposed approach are either visualized in heatmaps or listed as top words per topic by convention.

Note that for all but the Ich dataset, a heatmap has multiple columns and each column represents one topic learned. The number on top of each column is the learned Cox beta coefficient for that topic. Please refer to the paper for how to interpret topics and their heatmaps.

For the Ich dataset, we cannot produce the same heatmap as other datasets' because there are too many features. Instead, each heatmap (total of 5) represents one topic learned. Cox beta coefficients are denoted on the top. Features are ranked by deviation from the background word frequency, and only the top 100 words for each topic are shown.
