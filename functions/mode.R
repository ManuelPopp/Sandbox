require("dplyr")

mode <- function(..., ties = "first", digits = NA) {
  #' Get the mode value from a vector
  #' 
  #' Count the occurrences of values within a vector and return the most
  #' frequent one.
  #' 
  #' @param x A vector.
  #' @param ties Either 'first' to return the first value encountered in case
  #' of ties, or 'all', to return a vector of all most frequent values, instead.
  #' @param digits An integer. Number of digits to round floating point numbers
  #' to, before searching the mode value. Only used if set to a numeric value.
  #' The default is NA.
  #' 
  #' @examples
  #' mode(1, 2, "a", 2, "c", ties = "all")
  #' 
  #' values <- c(1.13, 1.02, 1.14, 1.51, 0.95, 1.12, 1.47, 0.69)
  #' mode(values, digits = 1)
  #' 
  #' @export
  
  # Get input values and keyword arguments
  args <- list(...)
  arg_names <- names(args)
  
  values = if (is.null(arg_names)) {
    do.call(c, args)
  } else {
    do.call(c, args[arg_names == "" | is.null(arg_names)])
  }
  
  kwargs = if (is.null(arg_names)) {
    list()
  } else {
    args[!is.null(arg_names) & arg_names != ""]
  }
  
  defaults <- list(ties = ties, digits = digits)
  params <- modifyList(defaults, kwargs)
  list2env(params, envir = environment())
  
  for (kwarg in names(kwargs)) {
    if (!kwarg %in% defaults) {
      warning(paste0("Unused keyword argument: ", kwarg))
    }
  }
  
  # Round floating point values
  if (is.numeric(values) & is.numeric(digits)) {
    values <- round(values, digits)
  }
  
  # Create frequency table
  freq_tab <- table(values)
  mode_values <- names(freq_tab)[which(freq_tab == max(freq_tab))]
  
  # Return mode
  ties <- tolower(ties)
  if (!ties %in% c("first", "all")) {
    stop(paste("Invalid value", ties, "for argument 'ties'."))
  }
  
  if (length(mode_values) > 1 & ties == "first") {
    warning(
      paste(
        "Multiple values with equal frequency found.",
        "Only the first one will be returned. Set ties='all' to return a",
        "vector instead."
        )
      )
    return(names(freq_tab)[which.max(freq_tab)])
  } else {
    return(names(freq_tab)[freq_tab == max(freq_tab)])
  }
}
