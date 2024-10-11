import <- function(...) {
  #' Import R packages. Install them if necessary.
  #' 
  #' @param ... any argument that can be passed to install.packages. Provide
  #' package names as character strings.
  #' @details The function installs only packages that are missing. Packages
  #' are loaded.
  #' @examples
  #' # Load packages
  #' import("dplyr", "MASS", "terra", dependencies = TRUE)
  #' 
  #' @seealso \code{\link[base]{install.packages}}
  #' @export
  args <- list(...)
  arg_names <- names(args)
  
  packages = if (is.null(arg_names)) {
    args
  } else {
    args[arg_names == "" | is.null(arg_names)]
  }
  
  kwargs = if (is.null(arg_names)) {
    list()
  } else {
    args[!is.null(arg_names) & arg_names != ""]
  }
  
  load <- function(package) {
    if (!requireNamespace(package, quietly = TRUE)) {
      do.call(install.packages, c(list(package), kwargs))
    }
    base::library(package, character.only = TRUE, pos = .GlobalEnv)
  }
  invisible(lapply(packages, load))
}
