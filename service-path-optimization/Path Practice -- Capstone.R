# Download data set with completed cases only:
completed <- read.csv("/Users/wendyguerrero/Desktop/QTM 220 /Lab Datasets/completed_pmdm.csv", stringsAsFactors = FALSE) 
fulldata <- read.csv("/Users/wendyguerrero/Desktop/QTM 220 /Lab Datasets/updated_datanew.csv", stringsAsFactors = FALSE) 

# Upload necessary packages:
library(dplyr)
library(ggplot2)
library(lubridate)
install.packages("igraph")
install.packages("networkD3")
library(igraph)
library(networkD3)
library(ggalluvial)

table(completed$LIVING_SITUATION_PRIOR_TO_ENTRY__C)
table(completed$LIVING_SITUATION_AT_ENTRY__C)
table(completed$LIVING_SITUATION_AT_EXIT__C)

# At least one column is filled out of the three we are looking into 
filtered_data <- subset(completed, 
                        !is.na(LIVING_SITUATION_PRIOR_TO_ENTRY__C) & LIVING_SITUATION_PRIOR_TO_ENTRY__C != "" |
                          !is.na(LIVING_SITUATION_AT_ENTRY__C) & LIVING_SITUATION_AT_ENTRY__C != "" |
                          !is.na(LIVING_SITUATION_AT_EXIT__C) & LIVING_SITUATION_AT_EXIT__C != "")
# Create a csv of the subset that has at least one column filled
write.csv(filtered_data, "filtered_completed_pmdm.csv", row.names = FALSE)

# Looking only at the cases that have ALL 3 columns filled
filtered_rows_count <- sum(!is.na(completed$LIVING_SITUATION_PRIOR_TO_ENTRY__C) & completed$LIVING_SITUATION_PRIOR_TO_ENTRY__C != "" &
                             !is.na(completed$LIVING_SITUATION_AT_ENTRY__C) & completed$LIVING_SITUATION_AT_ENTRY__C != "" &
                             !is.na(completed$LIVING_SITUATION_AT_EXIT__C) & completed$LIVING_SITUATION_AT_EXIT__C != "")

# Print the count [only 11 individual cases have all three rows filled]
print(filtered_rows_count)


filtered <- completed %>%
  filter(!is.na(LIVING_SITUATION_PRIOR_TO_ENTRY__C) & LIVING_SITUATION_PRIOR_TO_ENTRY__C != "" &
           !is.na(LIVING_SITUATION_AT_ENTRY__C) & LIVING_SITUATION_AT_ENTRY__C != "" &
           !is.na(LIVING_SITUATION_AT_EXIT__C) & LIVING_SITUATION_AT_EXIT__C != "")

links <- filtered %>%
  select(LIVING_SITUATION_PRIOR_TO_ENTRY__C, LIVING_SITUATION_AT_ENTRY__C, LIVING_SITUATION_AT_EXIT__C) %>%
  pivot_longer(cols = everything(), names_to = "Stage", values_to = "Living_Situation") %>%
  group_by(Stage, Living_Situation) %>%
  summarise(count = n(), .groups = "drop")

nodes <- data.frame(name = unique(links$Living_Situation))

links <- filtered %>%
  mutate(source1 = match(LIVING_SITUATION_PRIOR_TO_ENTRY__C, nodes$name) - 1,
         target1 = match(LIVING_SITUATION_AT_ENTRY__C, nodes$name) - 1,
         source2 = match(LIVING_SITUATION_AT_ENTRY__C, nodes$name) - 1,
         target2 = match(LIVING_SITUATION_AT_EXIT__C, nodes$name) - 1) %>%
  select(source1, target1, source2, target2) %>%
  pivot_longer(cols = everything(), values_to = "node") %>%
  mutate(next_node = lead(node)) %>%
  na.omit() %>%
  count(node, next_node, name = "value")

sankeyNetwork(Links = links, Nodes = nodes, 
              Source = "node", Target = "next_node", Value = "value",
              NodeID = "name", fontSize = 12, nodeWidth = 30)


ggplot(filtered, 
       aes(axis1 = LIVING_SITUATION_PRIOR_TO_ENTRY__C, 
           axis2 = LIVING_SITUATION_AT_ENTRY__C, 
           axis3 = LIVING_SITUATION_AT_EXIT__C)) +
  geom_alluvium(aes(fill = LIVING_SITUATION_PRIOR_TO_ENTRY__C), width = 1/12) +
  geom_stratum() +
  geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
  theme_minimal() +
  ggtitle("Living Situation Transition Pathways")

# To prevent overcrowding in plot

# Create a mapping of unique living situations to numeric values
unique_living_situations <- unique(c(filtered$LIVING_SITUATION_PRIOR_TO_ENTRY__C, 
                                     filtered$LIVING_SITUATION_AT_ENTRY__C, 
                                     filtered$LIVING_SITUATION_AT_EXIT__C))

# Assign numeric labels to each living situation
living_situation_map <- setNames(seq_along(unique_living_situations), unique_living_situations)

# Replace text values with numeric codes
filtered <- filtered %>%
  mutate(Prior = living_situation_map[LIVING_SITUATION_PRIOR_TO_ENTRY__C],
         Entry = living_situation_map[LIVING_SITUATION_AT_ENTRY__C],
         Exit  = living_situation_map[LIVING_SITUATION_AT_EXIT__C])

# Convert numeric values back to factors for plotting
filtered$Prior <- factor(filtered$Prior, levels = unique(filtered$Prior))
filtered$Entry <- factor(filtered$Entry, levels = unique(filtered$Entry))
filtered$Exit <- factor(filtered$Exit, levels = unique(filtered$Exit))

# Create the alluvial plot
ggplot(filtered, 
       aes(axis1 = Prior, 
           axis2 = Entry, 
           axis3 = Exit)) +
  geom_alluvium(aes(fill = Prior), width = 1/12) +
  geom_stratum() +
  geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
  theme_minimal() +
  ggtitle("Living Situation Transition Pathways") +
  scale_fill_manual(values = rainbow(length(unique_living_situations)), 
                    labels = unique_living_situations, 
                    name = "Living Situation")

# Print legend mapping numeric levels to living situations
print(data.frame(Level = seq_along(unique_living_situations), 
                 Living_Situation = unique_living_situations))

# If we only look at living situation at entry AND at exit rather than before entry
newdata <- completed %>%
  filter(!is.na(LIVING_SITUATION_AT_ENTRY__C) & LIVING_SITUATION_AT_ENTRY__C != "" &
           !is.na(LIVING_SITUATION_AT_EXIT__C) & LIVING_SITUATION_AT_EXIT__C != "")

links1 <- newdata %>%
  select(LIVING_SITUATION_AT_ENTRY__C, LIVING_SITUATION_AT_EXIT__C) %>%
  pivot_longer(cols = everything(), names_to = "Stage", values_to = "Living_Situation") %>%
  group_by(Stage, Living_Situation) %>%
  summarise(count = n(), .groups = "drop")

unique_living_situations1 <- unique(c(filtered$LIVING_SITUATION_AT_ENTRY__C, 
                                     filtered$LIVING_SITUATION_AT_EXIT__C))

# Assign numeric labels to each living situation
living_situation_map1 <- setNames(seq_along(unique_living_situations1), unique_living_situations1)

# Replace text values with numeric codes
newdata <- newdata %>%
  mutate(Entry = living_situation_map1[LIVING_SITUATION_AT_ENTRY__C],
         Exit  = living_situation_map1[LIVING_SITUATION_AT_EXIT__C])

# Convert numeric values back to factors for plotting
newdata$Entry <- factor(newdata$Entry, levels = unique(newdata$Entry))
newdata$Exit <- factor(newdata$Exit, levels = unique(newdata$Exit))

# Create the alluvial plot
newdata_clean <- newdata %>% drop_na(Entry, Exit)
ggplot(newdata_clean, 
       aes(axis1 = Entry, 
           axis2 = Exit)) +
  geom_alluvium(aes(fill = Entry), width = 1/12) +
  geom_stratum() +
  geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
  theme_minimal() +
  ggtitle("Living Situation Transition Pathways") +
  scale_fill_manual(values = rainbow(length(unique_living_situations1)), 
                    labels = unique_living_situations1, 
                    name = "Living Situation")

# Print legend mapping numeric levels to living situations
print(data.frame(Level = seq_along(unique_living_situations1), 
                 Living_Situation = unique_living_situations1))




