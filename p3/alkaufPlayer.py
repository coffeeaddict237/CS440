import sys
import math
# print to stderr for debugging purposes
# remove all debugging statements before submitting your code
msg = "Given board " + sys.argv[1] + "\n";
sys.stderr.write(msg);

#parse the input string, i.e., argv[1]

def check_triangles(possible_location, play):
    #will return a list of strings of losing colors
    #check 6 possible triangles and check which colors could cause a loss

    total_height = len(play) - 1

    height = total_height - int(last_play_location[1])
    print(height)
    dist_left = int(possible_location[2])
    dist_right = int(possible_location[3])
    #row = sys.argv[1][height]
    print(play)
    #colors for each of the surrounding locations    
    
    left_side = play[height][dist_left - 1]
    right_side = play[height][dist_right - 1] 
    upper_left = play[height - 1][dist_left - 1]
    upper_right = play[height - 1][dist_left]
    bottom_left = play[height + 1][dist_right]
    bottom_right = play[height + 1][dist_right - 1]
    
    losing_colors = []    
    
    if (left_side != "0" and upper_left != "0" and left_side != upper_left):
        colors = ["1", "2", "3"]
        colors.remove(left_side)
        colors.remove(upper_left)
        losing_colors.append(colors[0])
        
    if (upper_left != "0" and upper_right != "0" and upper_left != upper_right):
        colors = ["1", "2", "3"]
        colors.remove(upper_left)
        colors.remove(upper_right)
        if (colors[0] not in losing_colors):
            losing_colors.append(colors[0])
        
    if (upper_right != "0" and right_side != "0" and upper_right != right_side):
        colors = ["1", "2", "3"]
        colors.remove(upper_right)
        colors.remove(right_side)
        losing_colors.append(colors[0])
        if (colors[0] not in losing_colors):
            losing_colors.append(colors[0])
        
    if (right_side != "0" and bottom_right != "0" and right_side != bottom_right):
        colors = ["1", "2", "3"]
        colors.remove(right_side)
        colors.remove(bottom_right)
        losing_colors.append(colors[0])
        if (colors[0] not in losing_colors):
            losing_colors.append(colors[0])
        
    if (bottom_right != "0" and bottom_left != "0" and bottom_right != bottom_left):
        colors = ["1", "2", "3"]
        colors.remove(bottom_right)
        colors.remove(bottom_left)
        losing_colors.append(colors[0])
        if (colors[0] not in losing_colors):
            losing_colors.append(colors[0])
        
    if (bottom_left != "0" and left_side != "0" and bottom_left != left_side):
        colors = ["1", "2", "3"]
        colors.remove(bottom_left)
        colors.remove(left_side)
        losing_colors.append(colors[0])
        if (colors[0] not in losing_colors):
            losing_colors.append(colors[0])
    
    return(losing_colors)

#figure out depth that you can handle, progressive deepening    

def find_moves(current_location, play):
    #will return a list of strings of possible location
    
    possible_moves = []
    #"[13][302][1003][31002][100003][3000002][121212]LastPlay:(1,3,1,3)"
    #[[13], [302], ...]
    total_height = len(play)
    
    #last_play_location will look like [1, 3, 1, 3]
    #print(last_play_location)
    height = total_height - int(last_play_location[1])
    dist_left = int(last_play_location[2])
    dist_right = int(last_play_location[3])
    #row = sys.argv[1][height]
    
    #colors for each of the possible surrounding locations    
    #print(play[height][dist_left]) #should print the row that the last play was made
    left_side = (play[height])[dist_left - 1]
    right_side = play[height][dist_right - 1] 
    upper_left = play[height - 1][dist_left - 1]
    upper_right = play[height - 1][dist_left]
    bottom_left = play[height + 1][dist_right]
    bottom_right = play[height + 1][dist_right - 1]
    
    #check 6 surrounding positions
    
    # get position of left side 
    if (left_side == "0"):
        left_move = ["0", last_play_location[1], str(int(last_play_location[dist_left]) - 1), str(int(last_play_location[dist_right]) + 1)]
        possible_moves.append(left_move)
    
    #get position of right side
    if (right_side == "0"):
        right_move = ["0", last_play_location[1], str(int(last_play_location[dist_left]) + 1), str(int(last_play_location[dist_right]) - 1)]
        possible_moves.append(right_move)    
    
    if (upper_left == "0"):
        upper_left_move = ["0", str(int(last_play_location[height] - 1)), str(int(last_play_location[dist_left]) - 1), last_play_location[dist_right]]
        possible_moves.append(upper_left_move)
    
    if (upper_right == "0"):
        upper_right_move = ["0", str(int(last_play_location[height]) - 1), last_play_location[dist_left], str(int(last_play_location[dist_right]) - 1)]
        possible_moves.append(upper_right_move)
    
    if (bottom_left == "0"):
        bottom_left_move = ["0", str(int(last_play_location[height]) + 1), last_play_location[dist_left], str(int(last_play_location[dist_right]) + 1)]
        possible_moves.append(bottom_left_move)
    
    if (bottom_right == "0"):
        bottom_right_move = ["0", str(int(last_play_location[height]) + 1), str(int(last_play_location[dist_left]) + 1), last_play_location[dist_right]]
        possible_moves.append(bottom_right_move)  

    return possible_moves
    
    
    
#perform intelligent search to determine the next move

def minimax(current_move, depth, maximizingPlayer):
    
    if (depth == 0 or len(find_moves(current_move)) == []):
        
        #find heuristic value for this move (check colors, if you will lose, -10, if you win +10, if neither, 0)  
        triangle_check = check_triangles(current_move, play)
        if (len(triangle_check == 3) and maximizingPlayer):
            
            return (-10)
        
        elif (len(triangle_check == 3) and (not(maximizingPlayer))):        
        
            return (10)
        
        else:
            
            return (0)
    
    #If we're on a maximizing depth, max the possible value for you    
    
    if maximizingPlayer:
        
        best_value = -math.inf
        possible_moves = find_moves(current_move)
        
        for i in possible_moves:
            
            v = minimax(i, depth - 1, False)
            best_value = max(v, best_value)
            return best_value
    
    #If not, minimize the value (other player's turn)    
    
    else:
        
        best_value = math.inf
        possible_moves = find_moves(current_move)
        
        for i in possible_moves:
            
            v = minimax(i, depth - 1, True)
            best_value = min(v, best_value)
            return best_value
            
#sys.argv[1].split("[")           
play = sys.argv[1].split("]")

for i in range(len(play) - 1):
    play[i] = play[i][1:]
#play will look like this: ['13', '302', '1003', '30002', '100003', '3000002', '121212', 'LastPlay:(3,4,1,2)']
#print(play)
last_play = play[-1].split(":")

if(last_play[1] != "null"):
    last_play[-1] = last_play[-1][1:-1]
#print(last_play)
#print("Real last play:" + last_play)
play = play[0:-1]
print(play) #will print the board
total_rows = 0
for j in sys.argv[1]:
    if ("LastPlay" not in j):
        total_rows = total_rows + 1 
total_rows = total_rows - 1

if (last_play[-1] == "null"):
    #play can go anywhere, just choose a spot

    sys.stdout.write("(3,4,1,2)"); 

else:
            
    last_play_location = last_play[-1].split(",")
    possible_moves = find_moves(last_play_location, play) 
        
    if (possible_moves == []):
        #go anywhere (well, anywhere that's safe)
        open_moves = []
        #"[13][302][1003][31002][100003][3000002][121212]LastPlay:(1,3,1,3)"
        #[[13], [302], ...]
        for i in range(len(play)):
            if ("0") in play[i]:
                for c in range(len(play[i])):
                    if (play[i].index(c) == "0"):
                        new_move = ["0", str(total_rows - i), str(c), str(len(play[i]) - c - 1)]
        
    
    else:
        #if you only have one possible move, make it
        if (len(possible_moves) == 1):
            
            #check colors
            #put down color that is not wrong if possible
            losing_colors = check_triangles(possible_moves[0])
            
            if ((len(losing_colors) == 3)):
                #choose any of them, sorry, you've lost 
                sys.stdout.write(["1"] + possible_moves[1:]);
            
            elif ((len(losing_colors) == 2)):
                #one color is ok, you should probably choose that one
                colors = ["1", "2", "3"]
                colors.remove(losing_colors[0])
                colors.remove(losing_colors[1])
                
                sys.stdout.write(colors[0] + possible_moves[1:])
            
            else:
                
                make_move = possible_moves
                
                #choose a color
                colors = ["3", "1", "2"]
                find_color = check_triangles(make_move)
                for i in find_color:
                    if (i in colors):
                        colors.remove(i)
                
                if (colors != []):
                    good_color = colors[0]
                
                    final_move = colors[0] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
                
                else:
                    
                    final_move = ["1"] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));                                            
                
        else:
             
             #don't lose
             #do something
             #foreach possible move (up to 6 possible moves)
             #check triangles colors
             #check if dead end
             #check if moves are safe             
            depth = 5 
            total_moves = [] #all total moves
            triangle_checks = [] 
            losing_moves = []
            bad_moves = []    
            good_moves = []
            ok_moves = []
            minimax_scores = []
                           
            for i in range(len(possible_moves)):
                
                total_moves.append(possible_moves[i])
            
            
            #place any moves with failed triangle checks in "bad moves" list
            #for the rest of the moves, put each move through decision making algorithm
            
            for j in range(len(total_moves)):
                
                triangle_checks[j] = check_triangles(possible_moves[i], play)
                if (len(triangle_checks[j]) == 3):
                    
                    losing_moves.append(total_moves[j])
                    
                else:
                    #find the minimax scores for each potential move
                
                    minimax_scores[j] = minimax(total_moves[j], depth, True)
                    
            #check minimax scores for any 10 values, choose a "winning path", or avoid "losing path"
                
            for k in range(len(total_moves)):  
                
                if (minimax_scores[k] == 10):
                    
                    good_moves.append(total_moves[k])
                
                elif(minimax_scores[k] == 0):
                    
                    ok_moves.append(total_moves[k])
                    
                else:
                    
                    if (total_moves[k] not in losing_moves):
                        bad_moves.append(total_moves[k])
            
            if (good_moves != []):
                #choose a "winning path"
                      
                make_move = good_moves[0]
                
                #choose a color
                colors = ["3", "1", "2"]
                find_color = check_triangles(make_move, play)
                for i in find_color:
                    if (i in colors):
                        colors.remove(i)
                
                if (colors != []):
                    good_color = colors[0]
                
                    final_move = colors[0] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
                
                else:
                    
                    final_move = ["1"] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
                    
            
            elif (ok_moves != []):
                #choose a "safe path"
                make_move = ok_moves[0]
                #choose a color
                colors = ["1", "2", "3"]
                find_color = check_triangles(make_move)
                for i in find_color:
                    if (i in colors):
                        colors.remove(i)
                
                if (colors != []):
                    good_color = colors[0]
                
                    final_move = colors[0] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
                
                else:
                    
                    final_move = ["1"] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
            
            elif (bad_moves != []):
                #choose a "dangerous path"
                make_move = bad_moves[0]
                #choose a color
                colors = ["0", "1", "2"]
                find_color = check_triangles(make_move)
                for i in find_color:
                    if (i in colors):
                        colors.remove(i)
                
                if (colors != []):
                    good_color = colors[0]
                
                    final_move = colors[0] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
                
                else:
                    
                    final_move = ["1"] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
            
            else:
                #choose a "losing path" :(
                make_move = losing_moves[0]
                #choose a color
                colors = ["0", "1", "2"]
                find_color = check_triangles(make_move)
                for i in find_color:
                    if (i in colors):
                        colors.remove(i)
                
                if (colors != []):
                    good_color = colors[0]
                
                    final_move = colors[0] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));
                
                else:
                    
                    final_move = ["1"] + make_move[1:]
                    sys.stdout.write(str((final_move[0], final_move[1], final_move[2], final_move[3])));



#print to stdout for AtroposGame
#sys.stdout.write("(3,2,2,2)");
# As you can see Zook's algorithm is not very intelligent. He 
# will be disqualified.

