library(bnlearn)
library(Rgraphviz)

# Ground Truth Models for DETERMINISTIC environments, where r depends only on st and a

# ct prefix stands for Coffee Task
ct_go_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|suI][srJ|srI][swJ|suI:srI:swI][scJ|scI][reward|slI:scI]")
ct_gu_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|slI:suI][srJ|srI][swJ|swI][scJ|scI][reward|slI:suI:srI]")
ct_bc_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|suI][srJ|srI][swJ|swI][scJ|slI:scI][reward|slI:scI]")
ct_dc_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|suI][srJ|srI][swJ|swI][scJ][reward|slI:scI]")
ct_gt_models <- list("go" = ct_go_dbn, "gu" = ct_gu_dbn, "bc" = ct_bc_dbn, "dc" = ct_dc_dbn)
ct_actions <- c("go","gu","bc","dc")
ct_V <- c("slI", "suI", "srI", "swI", "scI", "slJ", "suJ", "srJ", "swJ", "scJ", "reward")
ct_t0.nodes <- c("slI", "suI", "srI", "swI", "scI")
ct_t1.nodes <- c("slJ", "suJ", "srJ", "swJ", "scJ", "reward")
ct_bl <- set2blacklist(ct_t0.nodes)
ct_bl <- rbind(ct_bl,set2blacklist(ct_t1.nodes))
ct_bl <- rbind(ct_bl, tiers2blacklist(list(ct_t0.nodes,ct_t1.nodes)))

# tt prefix stands for the two versions of the Taxi Task. The one with 3 relational variables and penalty on bad actions
tt_south_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI]")
tt_north_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI]")
tt_east_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI]")
tt_west_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI]")
tt_pick_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI:lI][lJ|lI][nwJ|nwI][reward|wpI:lI]")
tt_drop_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI:lI][lJ|lI][nwJ|nwI][reward|wpI:lI]")
tt_gt_models <- list("south" = tt_south_dbn,"north" = tt_north_dbn, "east" = tt_east_dbn, "west" = tt_west_dbn, "pick" = tt_pick_dbn, "drop" = tt_drop_dbn)
tt_actions <- c("south","north","east","west","pick","drop")
tt_V <- c("wpI", "lI", "nwI", "wpJ", "lJ", "nwJ", "reward")
tt_t0.nodes <- c("wpI", "lI", "nwI")
tt_t1.nodes <- c("wpJ", "lJ", "nwJ", "reward")
tt_bl <- set2blacklist(tt_t0.nodes)
tt_bl <- rbind(tt_bl,set2blacklist(tt_t1.nodes))
tt_bl <- rbind(tt_bl, tiers2blacklist(list(tt_t0.nodes,tt_t1.nodes)))

# Ground Truth Models for STOCHASTIC environments where r depends on st, st+1 and a

# ct2 prefix stands for Coffee Task
ct2_go_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|suI][srJ|srI][swJ|suI:srI:swI][scJ|scI][reward|slI:scI:slJ]")
ct2_gu_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|slI:suI][srJ|srI][swJ|swI][scJ|scI][reward|slI:suI:srI:suJ]")
ct2_bc_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|suI][srJ|srI][swJ|swI][scJ|slI:scI][reward|scI:scJ]")
ct2_dc_dbn <- model2network("[slI][suI][srI][swI][scI][slJ|slI][suJ|suI][srJ|srI][swJ|swI][scJ|scI][reward|slI:swI:scI:scJ]")
ct2_gt_models <- list("go" = ct2_go_dbn, "gu" = ct2_gu_dbn, "bc" = ct2_bc_dbn, "dc" = ct2_dc_dbn)
ct2_actions <- c("go","gu","bc","dc")
ct2_V <- c("slI", "suI", "srI", "swI", "scI", "slJ", "suJ", "srJ", "swJ", "scJ", "reward")
ct2_t0.nodes <- c("slI", "suI", "srI", "swI", "scI")
ct2_t1.nodes <- c("slJ", "suJ", "srJ", "swJ", "scJ")
ct2_t2.nodes <- c("reward")
ct2_bl <- set2blacklist(ct2_t0.nodes)
ct2_bl <- rbind(ct2_bl,set2blacklist(ct2_t1.nodes))
ct2_bl <- rbind(ct2_bl,set2blacklist(ct2_t2.nodes))
ct2_bl <- rbind(ct2_bl, tiers2blacklist(list(ct2_t0.nodes,ct2_t1.nodes)))
ct2_bl <- rbind(ct2_bl, tiers2blacklist(list(ct2_t0.nodes,ct2_t2.nodes)))
ct2_bl <- rbind(ct2_bl, tiers2blacklist(list(ct2_t1.nodes,ct2_t2.nodes)))

# tt2 prefix stands for the two versions of the Taxi Task. The one with 3 relational variables and penalty on bad actions
tt2_south_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI:nwJ]")
tt2_north_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI:nwJ]")
tt2_east_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI:nwJ]")
tt2_west_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI][lJ|lI:nwI][nwJ|nwI][reward|nwI:nwJ]")
tt2_pick_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI:lI][lJ|lI][nwJ|nwI][reward|wpI:lI]")
tt2_drop_dbn <- model2network("[wpI][lI][nwI][wpJ|wpI:lI][lJ|lI][nwJ|nwI][reward|wpI:wpJ:lI]")
tt2_gt_models <- list("south" = tt2_south_dbn,"north" = tt2_north_dbn, "east" = tt2_east_dbn, "west" = tt2_west_dbn, "pick" = tt2_pick_dbn, "drop" = tt2_drop_dbn)
tt2_actions <- c("south","north","east","west","pick","drop")
tt2_V <- c("wpI", "lI", "nwI", "wpJ", "lJ", "nwJ", "reward")
tt2_t0.nodes <- c("wpI", "lI", "nwI")
tt2_t1.nodes <- c("wpJ", "lJ", "nwJ")
tt2_t2.nodes <- c("reward")
tt2_bl <- set2blacklist(tt2_t0.nodes)
tt2_bl <- rbind(tt2_bl,set2blacklist(tt2_t1.nodes))
tt2_bl <- rbind(tt2_bl,set2blacklist(tt2_t2.nodes))
tt2_bl <- rbind(tt2_bl, tiers2blacklist(list(tt2_t0.nodes,tt2_t1.nodes)))
tt2_bl <- rbind(tt2_bl, tiers2blacklist(list(tt2_t0.nodes,tt2_t2.nodes)))
tt2_bl <- rbind(tt2_bl, tiers2blacklist(list(tt2_t1.nodes,tt2_t2.nodes)))


task_parameters <- list("CoffeeTaskEnv" = list("deterministic" = (list("gt_models" = ct_gt_models, "actions" = ct_actions, "V" = ct_V, "t0.nodes" = ct_t0.nodes, "t1.nodes" = ct_t1.nodes, "bl" = ct_bl)),
                                               "stochastic" = (list("gt_models" = ct2_gt_models, "actions" = ct2_actions, "V" = ct2_V, "t0.nodes" = ct2_t0.nodes, "t1.nodes" = ct2_t1.nodes, "t2.nodes" = ct2_t2.nodes, "bl" = ct2_bl))
                                                                      ),
                       "TaxiBigEnv"    = list("deterministic" = (list("gt_models" = tt_gt_models, "actions" = tt_actions, "V" = tt_V, "t0.nodes" = tt_t0.nodes, "t1.nodes" = tt_t1.nodes, "bl" = tt_bl)),
                                           "stochastic" = (list("gt_models" = tt2_gt_models, "actions" = tt2_actions, "V" = tt2_V, "t0.nodes" = tt2_t0.nodes, "t1.nodes" = tt2_t1.nodes, "t2.nodes" = ct2_t2.nodes, "bl" = tt2_bl))
                                                                   ),
                       "TaxiSmallEnv"   = list("deterministic" = (list("gt_models" = tt_gt_models, "actions" = tt_actions, "V" = tt_V, "t0.nodes" = tt_t0.nodes, "t1.nodes" = tt_t1.nodes, "bl" = tt_bl)),
                                               "stochastic" = (list("gt_models" = tt2_gt_models, "actions" = tt2_actions, "V" = tt2_V, "t0.nodes" = tt2_t0.nodes, "t1.nodes" = tt2_t1.nodes, "t2.nodes" = ct2_t2.nodes, "bl" = tt2_bl))
                                                                      )
                       )

plot_ground_truths <- function(environment_name, environment_type, gt_folder){

  params = task_parameters[[environment_name]][[environment_type]]

  jpeg(paste(gt_folder,"/ground_truth_models.jpg", sep = ""), width = 1920, height = 1080)

  actions_number = length(params$gt_models)

  par(mfrow=c(1,actions_number))

  for (action_name in params$actions){

    name <- paste("DBN for action", action_name)
    gR <- graphviz.plot(params$gt_models[[action_name]], render = FALSE, main = name)
    sg0 <- list(graph = subGraph(params$t0.nodes, gR), cluster = TRUE)
    sg1<- list(graph = subGraph(params$t1.nodes, gR), cluster = TRUE)
    if(environment_type == "stochastic"){
      sg2 <- list(graph = subGraph(params$t2.nodes, gR), cluster = TRUE)
    }
    subGList <- list(sg0,sg1)
    if (environment_type == "stochastic"){
      subGList <- append(subGList, list(sg2))
    }
    gR <- layoutGraph(gR, attrs = list(graph = list(rankdir = "LR")), subGList = subGList)

    cross <- vector()

    # Draw arcs from t0 to t1
    for (ori in params$t0.nodes){
      for (dest in params$t1.nodes){
        cross <- append(cross,paste(ori, "~", dest, sep = ""))
      }

      if(environment_type == "stochastic"){ # Draw arcs from t0 to t2
        for (dest in params$t2.nodes){
          cross <- append(cross,paste(ori, "~", dest, sep = ""))
        }

      }

    }

    if(environment_type == "stochastic"){ # Draw arcs from t1 to t2
      for (ori in params$t1.nodes){
        for (dest in params$t2.nodes){
          cross <- append(cross,paste(ori, "~", dest, sep = ""))
        }

      }
    }

    edgeRenderInfo(gR)$col[cross] <- "red"
    renderGraph(gR)


  }

  dev.off()
}

dbn_inference <- function (environment_name, environment_type, current_state, dbn_fit){

  params <- task_parameters[[environment_name]][[environment_type]]

  # FOR DEBUF ONLY: First, load the corresponding DBN model in the specified path
  # dbn_fit <- read.net(dbn_fit, debug = FALSE)

  #Note that both cpquery and cpdist are based on Monte Carlo particle filters, and therefore they may return slightly different values on different runs.
  #You can reduce the variability in the inference runs by increasing the number of draws in the sampling procedure by using the tuning parameter

  variable_names = params$t0.nodes
  variable_values = current_state
  parents <- intersect(dbn_fit$reward$parents, variable_names)

  names(variable_values) <- variable_names

  # Thi is the way whe using fixed values
  # prob <- cpquery(dbn.fit, event = (reward == 1),
  #           evidence = (slI == 0) & (suI == 0) &
  #             (srI == 0) &  (swI == 0) &
  #              (scI == 0) & (shI == 0))

  set.seed(1)

  # This programmatically build a conditional probability query...
  # qtxt <- paste("cpquery(dbn.fit, ", "event = (reward == 1)", ", evidence = (", "(slI == ", slI, ")", "&", "(suI == ", suI, ")",
  #              "&", "(srI == ", srI, ")","&", "(swI == ", swI, ")","&", "(scI == ", scI, ")",
  #              "&", "(shI == ", shI, ")","))")

  # This is the first part. Change the reward == 0
  #text_query <- "cpquery(dbn.fit, event = (reward == 0), evidence = ("

  #text_query <- "cpquery(dbn.fit, event = (reward == 2) , evidence = ("

  text_query <- ""

  if(length(parents) != 0) {
    text_query <- "cpdist(dbn_fit, 'reward' , evidence = ("
    # Then, construct the evidence based on the parents of reward only
    for (var in parents){
      text_query <- paste(text_query, "(", var, "==", variable_values[[var]], ") & ")
    }
    # Remove the last & and space
   text_query <- substring(text_query, 1, nchar(text_query) - 2)
   # Finally close the brackets
   text_query <- paste(text_query, "))")
    prob <- prop.table(table(eval(parse(text=text_query))))
  }
  else{
    # The reward variable is independent of every variable so we dont need to add evidence
    #text_query <- "cpdist(dbn_fit, 'reward')"
    prob <- dbn_fit$reward$prob
  }

  # This way does not work
  # prob <- cpquery(dbn.fit, event = (reward == "1"),
  #           evidence = (slI == scausal_based_action_selection_2ubstring(current_state,1,1)) & (suI == substring(current_state,2,2)) &
  #             (srI == substring(current_state,3,3)) &  (swI == substring(current_state,4,4)) &
  #              (scI == substring(current_state,5,5)) & (shI == substring(current_state,6,6)))

  return (prob)

}

load_model <- function (causal_model){
  dbn.fit <- read.net(causal_model, debug = FALSE)
  return (dbn.fit)
}

causal_discovery_using_rl_data <- function(environment_name, environment_type, data_set_path, action_name){

      params = task_parameters[[environment_name]][[environment_type]]

      # Filter the data frame to obtain variables of interest
      Data <- read.table(paste(data_set_path,action_name,".txt", sep=""), header = TRUE, colClasses = "factor")[,params$V]

      # TODO Make a verification for cardinality here before to use HC
      # Manual check for all variables to have at least 2 values in Data
      # count  = apply(Data, 2, function(x) length(unique(x)))
      #
      # CD_posible = TRUE
      # for (column in count){
      #    if(column[1] != 2){
      #      CD_posible = FALSE
      #      break;
      #    }
      # }

     # Learn the models using the blacklist and highclimbing algorithm
     #  if(CD_posible){ # Por ahora siempre va a ser posible el descubrimiento por la inicializacion de los datos
        learned <- tabu(Data,blacklist = params$bl)
        fitted <- bn.fit(learned,Data)
        write.net(paste(data_set_path,action_name,".net", sep=""), fitted)
      # }else{
        fitted <- "null"
        # TODO implementar esto
      # }

      ground_truth_model = params$gt_models[[action_name]]

      # if(action_name == 'go'){
      #   ground_truth_model = go_dbn
      # }
      # else if(action_name == 'gu'){
      #   ground_truth_model = gu_dbn
      # }
      # else if(action_name == 'bc'){
      #   ground_truth_model = bc_dbn
      # }
      # else if(action_name == 'dc'){
      #   ground_truth_model = dc_dbn
      # }

      dist <- shd(ground_truth_model, learned)

      comparisson <- compare(ground_truth_model, learned, arcs = TRUE)
      tp <- comparisson[[1]]
      fp <- comparisson[[2]]
      fn <- comparisson[[3]]

      # precision = dim(tp)[1]/(dim(tp)[1] + dim(fp)[1])
      # recall =  dim(tp)[1]/(dim(tp)[1] + dim(fn)[1])
      # f1 <- (2 * precision * recall)/(precision + recall)
      # dist <-  f1

      jpeg(paste(data_set_path,action_name,".jpg", sep=""), width = 1920, height = 1080)
      par(mfrow = c(1, 2))
      graphs <- graphviz.compare(ground_truth_model, learned,  main = c("Ground Truth", "Discovered"),
      sub = paste("SHD =", c("0", dist)))

      gT <- as.graphNEL(ground_truth_model)
      sg0 <- list(graph = subGraph(params$t0.nodes, gT), cluster = TRUE)
      sg1<- list(graph = subGraph(params$t1.nodes, gT), cluster = TRUE)
      if (environment_type == "deterministic"){
        gT <- layoutGraph(gT, attrs = list(graph = list(rankdir = "LR")), subGList = list(sg0,sg1))
      }
      if (environment_type == "stochastic"){
        sg2<- list(graph = subGraph(params$t2.nodes, gT), cluster = TRUE)
        gT <- layoutGraph(gT, attrs = list(graph = list(rankdir = "LR")), subGList = list(sg0,sg1,sg2))
      }

      renderGraph(gT)

      gR <- graphs[[2]]
      sg3 <- list(graph = subGraph(params$t0.nodes, gR), cluster = TRUE)
      sg4<- list(graph = subGraph(params$t1.nodes, gR), cluster = TRUE)
      if (environment_type == "deterministic"){
        gR <- layoutGraph(gR, attrs = list(graph = list(rankdir = "LR")), subGList = list(sg3,sg4))
      }
      if (environment_type == "stochastic"){
            sg5<- list(graph = subGraph(params$t2.nodes, gR), cluster = TRUE)
            gR <- layoutGraph(gR, attrs = list(graph = list(rankdir = "LR")), subGList = list(sg3,sg4,sg5))
      }

      tp_arc <- vector()
      fp_arc <- vector()
      fn_arc <- vector()
      if(dim(tp)[1] > 0){
        for (i in 1:dim(tp)[1]){
        tp_arc <- append(tp_arc,paste(tp[i,][[1]], "~", tp[i,][[2]], sep = ""))
      }
      }
      if(dim(fp)[1] > 0){
         for (i in 1:dim(fp)[1]){
            fp_arc <- append(fp_arc,paste(fp[i,][[1]], "~", fp[i,][[2]], sep = ""))
          }
      }
      if(dim(fn)[1] > 0){
        for (i in 1:dim(fn)[1]){
            fn_arc <- append(fn_arc,paste(fn[i,][[1]], "~", fn[i,][[2]], sep = ""))
          }
      }
      edgeRenderInfo(gR)$col[tp_arc] <- "green"
      edgeRenderInfo(gR)$col[fp_arc] <- "red"
      edgeRenderInfo(gR)$col[fn_arc] <- "yellow"

      renderGraph(gR)
      dev.off()

      output <- list(paste(data_set_path,action_name,".net", sep=""), dist)

      return (output)

}

# p <- plot_ground_truths("CoffeeTaskEnv","stochastic", "D:/000-Code/Python/carl/ground_truth_models/our_gym_environments/CoffeeTaskEnv-v0")
# d <- dbn_inference("CoffeeTaskEnv", "deterministic", c(0,0,0,0,0),"D:/000-Code/Python/carl/experiments_results/Deterministic Ground Truths Coffee Task/trial 1/cd_data_and_results/CRL-T60_GO/0_10/1980/go.net")
# d <- dbn_inference("OurTaxi2Env", c(1,2,5),"D:/000-Code/Python/causal_rl/experiments_results/20221108 103846 OurTaxi2Env T = 200 E = 1000 t = 0.7 trials = 1/trial 1/cd_data/0_37/200/pick.net")
# causal_discovery_using_rl_data("TaxiSmallEnv","deterministic","/home/kimo/PycharmProjects/carl/20-TaxiTask/20230606 180210 TaxiSmallEnv-deterministic RL vs RLforCD T20 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_98/20/", "pick")