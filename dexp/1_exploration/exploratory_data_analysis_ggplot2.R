head(diamonds)
summary(diamonds)
str(diamonds)

ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) + geom_point()

ggplot(diamonds, aes(x = carat, y = price)) + geom_point()

ggplot(diamonds, aes(x = log(carat), y = log(price))) + geom_point()

ggplot(diamonds[sample(nrow(diamonds), 10000), ], aes(x = carat, y = price)) + geom_point()


nrow(diamonds)

ggplot(diamonds, aes(x = carat, y = x * y * z)) + geom_point()

set.seed(639245)
dsmall <- diamonds[sample(nrow(diamonds),500),] 
summary(dsmall)
summary(diamonds)

ggplot(dsmall, aes(x = carat, y = price, color = color)) + geom_point()

ggplot(dsmall, aes(x = carat, y = price, shape = cut)) + geom_point()

ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 1/10)

ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 1/100)

ggplot(diamonds, aes(x = carat, y = price)) + geom_point(alpha = 1/200)

ggplot(dsmall, aes(x = carat, y = price)) + geom_line() + geom_smooth()


ggplot(dsmall, aes(x = carat, y = price)) + geom_point() + geom_smooth()

ggplot(dsmall, aes(x = carat, y = price)) + geom_point() + geom_smooth(span = 0.1)

ggplot(diamonds, aes(x = cut)) + geom_bar()


my_plot = ggplot(diamonds, aes(x = cut)) +  geom_bar()
print(my_plot) # Display the initial bar chart
my_plot = my_plot + geom_text(stat='count', aes(label=..count..), vjust=-1)
print(my_plot) # Display counts as labels on each bar
my_plot = my_plot +   theme(text = element_text(size = 14))
print(my_plot) # Display the plot with the new font size
my_plot = my_plot + ylim(0, 35000)
print(my_plot) # Display the plot with the modified y-axis limit
my_plot = my_plot + geom_bar(fill = 'lightsteelblue')
print(my_plot) # Display the plot with the new bar color
my_plot = my_plot + coord_flip()
print(my_plot) # Display the final plot with flipped axes

hist(diamonds$price) 

ggplot(diamonds, aes(x = price)) + geom_histogram(binwidth = 500)

ggplot(diamonds, aes(x = price)) + geom_histogram(bins = 10)

ggplot(diamonds, aes(x = price)) + geom_density()

ggplot(diamonds, aes(x = price)) + geom_density(fill = 'lightsteelblue4')


my_plot = ggplot(diamonds, aes(x = cut, fill = clarity)) + geom_bar() #second dimension of color within cut 
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

ggplot(mpg, aes(displ, hwy, color = factor(cyl)))+  geom_point()  + stat_smooth(method = "lm")

ggplot(diamonds, aes(x = cut, y = price, fill = cut)) + geom_boxplot() + scale_fill_brewer(palette = "Set3") + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ggtitle("Boxplot of Price by Cut")

ggplot(diamonds, aes(x = carat, y = price, color = cut)) + geom_point(alpha = 0.6) + geom_smooth(method = "lm", se = FALSE, linetype = "dashed") + theme_minimal() + ggtitle("Price vs Carat with Trend Line")

ggplot(diamonds, aes(x = price, fill = cut)) + geom_density(alpha = 0.6) + scale_fill_brewer(palette = "Paired") + ggtitle("Density Plot of Price by Cut")

ggplot(diamonds, aes(x = price)) + geom_histogram(bins = 30, fill = "skyblue", color = "black") + facet_grid(cut ~ clarity) + ggtitle("Distribution of Price by Cut and Clarity")

ggplot(diamonds, aes(x = cut, y = color, fill = carat)) +  geom_tile() +  scale_fill_gradient(low = "white", high = "blue") +  ggtitle("Heatmap of Cut and Color vs Carat")

ggplot(diamonds, aes(x = cut, y = price, fill = clarity)) + geom_violin() + coord_flip() + scale_fill_brewer(palette = "Set2") + ggtitle("Violin Plot of Price by Cut and Clarity")

ggplot(dsmall, aes(x = carat, y = price, size = depth, color = cut)) + geom_point(alpha = 0.6) + scale_size_continuous(range = c(2, 10)) + ggtitle("Price vs Carat with Point Size by Depth")




