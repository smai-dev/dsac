# ggplot2 examples: exploratory views of built-in R datasets.
# Code transformations and plotting settings are preserved.
library(ggplot2)

# Review records or summary information before visual interpretation.
head(diamonds)
# Review records or summary information before visual interpretation.
summary(diamonds)
# Review records or summary information before visual interpretation.
str(diamonds)

# Examine a selected relationship or distribution using ggplot2.
ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) + geom_point()

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = carat, y = price)) + geom_point()

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = log(carat), y = log(price))) + geom_point()

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds[sample(nrow(diamonds), 10000), ], aes(x = carat, y = price)) + geom_point()


nrow(diamonds)

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = carat, y = x * y * z)) + geom_point()

# Make the subsequent random sample reproducible.
set.seed(639245)
dsmall <- diamonds[sample(nrow(diamonds),500),] 
# Review records or summary information before visual interpretation.
summary(dsmall)
# Review records or summary information before visual interpretation.
summary(diamonds)

# Examine a selected relationship or distribution using ggplot2.
ggplot(dsmall, aes(x = carat, y = price, color = color)) + geom_point()

# Examine a selected relationship or distribution using ggplot2.
ggplot(dsmall, aes(x = carat, y = price, shape = cut)) + geom_point()

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 1/10)

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 1/100)

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 1/200)

# Examine a selected relationship or distribution using ggplot2.
ggplot(dsmall, aes(x = carat, y = price)) + geom_line() + geom_smooth()


# Examine a selected relationship or distribution using ggplot2.
ggplot(dsmall, aes(x = carat, y = price)) + geom_point() + geom_smooth()

# Examine a selected relationship or distribution using ggplot2.
ggplot(dsmall, aes(x = carat, y = price)) + geom_point() + geom_smooth(span = 0.1)

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = cut)) + geom_bar()


my_plot = ggplot(diamonds, aes(x = cut)) +  geom_bar()
print(my_plot)
my_plot = my_plot + geom_text(stat='count', aes(label=..count..), vjust=-1)
print(my_plot)
my_plot = my_plot +   theme(text = element_text(size = 14))
print(my_plot)
my_plot = my_plot + ylim(0, 35000)
print(my_plot)
my_plot = my_plot + geom_bar(fill = 'lightsteelblue')
print(my_plot)
my_plot = my_plot + coord_flip()
print(my_plot)

# Inspect a univariate distribution using base R.
hist(diamonds$price) 

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = price)) + geom_histogram(binwidth = 500)

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = price)) + geom_histogram(bins = 10)

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = price)) + geom_density()

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = price)) + geom_density(fill = 'lightsteelblue4')


my_plot = ggplot(diamonds, aes(x = cut, fill = clarity)) + geom_bar()
my_plot


my_plot = ggplot(diamonds, aes(x = color, fill = cut)) + geom_bar()
my_plot

my_plot = ggplot(diamonds, aes(x = price, fill = cut)) + geom_histogram(binwidth = 500) 
my_plot

my_plot = ggplot(diamonds, aes(x = price, color = cut)) + geom_density()
my_plot

my_plot = ggplot(diamonds, aes(x = cut, y = price)) + geom_boxplot() + coord_flip() + ggtitle("Boxplots of Price by Cut")
my_plot

my_plot = ggplot(diamonds, aes(x = color, y = price / carat)) + geom_jitter()
my_plot

# Examine a selected relationship or distribution using ggplot2.
ggplot(mpg, aes(displ, hwy, color = factor(cyl)))+  geom_point()  + stat_smooth(method = "lm")

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = cut, y = price, fill = cut)) + geom_boxplot() + scale_fill_brewer(palette = "Set3") + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ggtitle("Boxplot of Price by Cut")

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = carat, y = price, color = cut)) + geom_point(alpha = 0.6) + geom_smooth(method = "lm", se = FALSE, linetype = "dashed") + theme_minimal() + ggtitle("Price vs Carat with Trend Line")

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = price, fill = cut)) + geom_density(alpha = 0.6) + scale_fill_brewer(palette = "Paired") + ggtitle("Density Plot of Price by Cut")

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = price)) + geom_histogram(bins = 30, fill = "skyblue", color = "black") + facet_grid(cut ~ clarity) + ggtitle("Distribution of Price by Cut and Clarity")

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = cut, y = color, fill = carat)) +  geom_tile() +  scale_fill_gradient(low = "white", high = "blue") +  ggtitle("Heatmap of Cut and Color vs Carat")

# Examine a selected relationship or distribution using ggplot2.
ggplot(diamonds, aes(x = cut, y = price, fill = clarity)) + geom_violin() + coord_flip() + scale_fill_brewer(palette = "Set2") + ggtitle("Violin Plot of Price by Cut and Clarity")

# Examine a selected relationship or distribution using ggplot2.
ggplot(dsmall, aes(x = carat, y = price, size = depth, color = cut)) + geom_point(alpha = 0.6) + scale_size_continuous(range = c(2, 10)) + ggtitle("Price vs Carat with Point Size by Depth")




