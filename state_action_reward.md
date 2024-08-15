# state
- direction of snake head
- direction of food from snake head - 4 directions
- snake array (0 for nothing, 1 for snake) (uspe CNN in model)
- obstacle array
- position of snake head - HOW? - value is 5 in snake_array

# rewards
- +1 for moving closer to apple
- -1 for moving away from apple
- +10 for eating apple
- -100 for dying

# actions
- straight
- left turn
- right turn
