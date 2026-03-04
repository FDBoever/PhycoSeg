# Load libraries
suppressPackageStartupMessages({
  library(EBImage)
  library(tidyverse)
  library(cowplot)

})

# === 1. Load image ===
img_path <- "/Users/frederikdeboever/DATA/fabrics/Growth2025/pictures/11-07-25/WhatsApp Image 2025-07-12 at 08.31.53_masked.png"
img_path <- "/Users/frederikdeboever/DATA/fabrics/Growth2025/pictures/11-07-25/WhatsApp Image 2025-07-12 at 08.31.52.png"
img_path <- "/Users/frederikdeboever/DATA/fabrics/Growth2025/pictures/14-07-2025/tank4_2025-07-14 at 13.19.29.jpeg"

img_path <- "/Users/frederikdeboever/DATA/fabrics/Growth2025/pictures/14-07-2025/tank6_2025-07-14 at 13.19.25 2.png"
img_path <- "/Users/frederikdeboever/DATA/fabrics/Growth2025/pictures/14-07-2025/tank6_2025-07-14 at 13.19.24.jpeg"


img_dir <- "/Users/frederikdeboever/DATA/fabrics/Growth2025/pictures/14-07-2025/"

png_files <- list.files(path = img_dir, pattern = "\\.png$", full.names = TRUE)

all_features <- c()
for (img_path in png_files) {
  img <- EBImage::readImage(img_path)
  EBImage::display(img,method='raster')


  # Step 1: Binary segmentation (RGBA support)
  if (numberOfFrames(img) == 4) {
    alpha <- img[,,4]
    binary <- alpha > 0
  } else {
    gray <- channel(img, "gray")
    thresh <- otsu(gray)
    binary <- gray < thresh  # Seaweed is darker than background
  }

  #colorMode(img) <- "Grayscale"
  EBImage::display(img,method='raster')
  EBImage::display(binary,method='raster')


  binary_clean <- EBImage::fillHull(binary)
  binary_clean <- EBImage::opening(binary_clean, makeBrush(1, shape = 'disc'))
  binary_clean <- EBImage::closing(binary_clean, makeBrush(1, shape = 'disc'))
  binary_clean_img <- EBImage::Image(binary_clean, colormode = "Grayscale")
  EBImage::display(binary_clean_img, method = 'raster')

  #binary_clean_img <- Image(binary_clean, colormode = "Grayscale")
  EBImage::display(binary_clean,method='raster')
  EBImage::display(binary_clean_img,method='raster')


  #labelling objects
  labeled <- EBImage::bwlabel(binary_clean)
  EBImage::display(labeled,method='raster')

  #remove small objects
  min_px <- 500
  size_table <- table(labeled)
  remove_ids <- as.numeric(names(size_table[size_table < min_px]))
  labeled[labeled %in% remove_ids] <- 0
  labeled <- bwlabel(labeled)  # relabel after removing

  labeled_2d <-  EBImage::Image(labeled, colormode = "Grayscale")
  img_2d <- img[,,1]
  EBImage::display(labeled,method='raster')
  EBImage::display(labeled_2d,method='raster')

  features_shape   <- computeFeatures.shape(labeled_2d)
  features_moment  <- computeFeatures.moment(labeled_2d)
  features_basic   <- computeFeatures.basic(labeled_2d, img_2d)
  features_texture <- computeFeatures.haralick(labeled_2d, img_2d) # Haralick texture features
  features <- cbind(features_shape, features_moment, features_basic, features_texture)
  features <- as.data.frame(features)
  features$label <- as.integer(rownames(features))  # Extract label number

  # Add derived features
  features$complexity <- (features$s.perimeter^2) / (4 * pi * features$s.area)

  # Add filename column
  features$filename <- basename(img_path)
  # Get the prefix before the first underscore
  features$prefix <- strsplit(basename(img_path), "_")[[1]][1]

  # Add to master data frame
  all_features <- rbind(all_features, features)
}


scaling <- rbind(c('tank9_2025-07-14 at 13.19.20 2.png', '650.44','15','B'),
      c('tank9_2025-07-14 at 13.19.19 (1) 2.png', '662.03','15','B'),
      c('tank8_2025-07-14 at 13.19.20 (1) 2.png', '636.01','15','B'),
      c('tank7_2025-07-14 at 13.19.22 2.png', '680','15','B'),
      c('tank7_2025-07-14 at 13.19.22 (1) 2.png', '716','15','B'),
      c('tank6_2025-07-14 at 13.19.25 2.png', '622.23','15','A'),
      c('tank5_2025-07-14 at 13.19.26 2.png','600.08','15','A'),
      c('tank5_2025-07-14 at 13.19.27 2.png','692','15','A'),
      c('tank4_2025-07-14 at 13.19.32 2.png','698.07','15','A'),
      c('tank4_2025-07-14 at 13.19.29 2.png','652.03','15','A'),
      c('tank6_2025-07-14 at 13.19.25.png', '632.03','15','A')) %>% data.frame()
colnames(scaling) <- c('filename','px','cm','batch')
scaling$px <- as.numeric(as.character(scaling$px))
scaling$cm <- as.numeric(as.character(scaling$cm))

scaling$pixels_per_cm <- scaling$px / scaling$cm
scaled_df <- merge(all_features, scaling, by = "filename")


scaled_df$s.area_cm2       <- scaled_df$s.area / (scaled_df$pixels_per_cm^2)
scaled_df$s.perimeter_cm   <- scaled_df$s.perimeter / scaled_df$pixels_per_cm
scaled_df$s.radius.mean_cm <- scaled_df$s.radius.mean / scaled_df$pixels_per_cm
scaled_df$s.radius.sd_cm   <- scaled_df$s.radius.sd / scaled_df$pixels_per_cm
scaled_df$s.radius.min_cm  <- scaled_df$s.radius.min / scaled_df$pixels_per_cm
scaled_df$s.radius.max_cm  <- scaled_df$s.radius.max / scaled_df$pixels_per_cm
scaled_df$m.majoraxis_cm   <- scaled_df$m.majoraxis / scaled_df$pixels_per_cm


p.area <- scaled_df %>%
  filter(s.area_cm2 < 100) %>%
  ggplot(aes(prefix, s.area_cm2)) +
  geom_boxplot(aes(fill=batch),alpha=.7) +
  geom_jitter(width = .1,shape=21, aes(fill=batch))+
  scale_fill_manual(values = c("firebrick", "grey")) +
  theme_classic(base_size = 10) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(color = "black", size = 10, face = "plain"),
    axis.text.x = element_text(angle=45,hjust=1),
    axis.line = element_line(color = "black", size = 0.4),
    axis.ticks = element_line(color = "black", size = 0.4),
    axis.text = element_text(color = "black"),
    legend.position = "none",  # or "top" with small text if needed
    plot.margin = margin(5, 5, 5, 5))+ labs(x = "Tank", y = expression("Area (cm"^2*")"), color = "Tank")

p.perimeter <- scaled_df %>%
  filter(s.area_cm2 < 100) %>%
  ggplot(aes(prefix, s.perimeter_cm)) +
  geom_boxplot(aes(fill=batch),alpha=.7) +
  geom_jitter(width = .1,shape=21, aes(fill=batch))+
  scale_fill_manual(values = c("firebrick", "grey")) +
  theme_classic(base_size = 10) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(color = "black", size = 10, face = "plain"),
    axis.text.x = element_text(angle=45,hjust=1),
    axis.line = element_line(color = "black", size = 0.4),
    axis.ticks = element_line(color = "black", size = 0.4),
    axis.text = element_text(color = "black"),
    legend.position = "none",  # or "top" with small text if needed
    plot.margin = margin(5, 5, 5, 5))+ labs(x = "Tank", y = expression("Perimeter (cm)"), color = "Tank")


p.area.perimeter <- scaled_df %>%
  filter(s.area_cm2 < 100) %>%
  ggplot(aes(s.area_cm2, s.perimeter_cm)) +
  #geom_boxplot(aes(fill=batch),alpha=.7) +
  geom_point(shape=21, aes(fill=batch)) +
  geom_smooth(method='lm', aes(group=batch,fill=batch,color=batch),size = 0.5) +
  scale_fill_manual(values = c("firebrick", "grey")) +
  scale_color_manual(values = c("firebrick", "grey")) +
  theme_classic(base_size = 10) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(color = "black", size = 10, face = "plain"),
    axis.line = element_line(color = "black", size = 0.4),
    axis.ticks = element_line(color = "black", size = 0.4),
    axis.text = element_text(color = "black"),
    legend.position = "none",  # or "top" with small text if needed
    plot.margin = margin(5, 5, 5, 5))+
  labs(x = expression("Area (cm"^2*")"), y = expression("Perimeter (cm)"), color = "Tank")



#(p.area + p.perimeter + p.area.perimeter) +
#  patchwork::plot_layout(heights = c(1,1,2))
p.combined <- (p.area + p.perimeter + p.area.perimeter) +
  patchwork::plot_layout(widths = c(1, 1, 2))


ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_area_v_perimeter_',format(Sys.Date(), "%m.%d.%y"),'.png'),
                plot= p.combined,
                height=3.5, width=8, dpi=300)

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_area_v_perimeter_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.combined,
                height=3.5, width=8)


p.ecc <- scaled_df %>%
  filter(s.area_cm2 < 100) %>%
  ggplot(aes(prefix, m.eccentricity)) +
  geom_boxplot(aes(fill=batch),alpha=.7) +
  geom_jitter(width = .1,shape=21, aes(fill=batch))+
  scale_fill_manual(values = c("firebrick", "grey")) +
  theme_classic(base_size = 10) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(color = "black", size = 10, face = "plain"),
    axis.text.x = element_text(angle=45,hjust=1),
    axis.line = element_line(color = "black", size = 0.4),
    axis.ticks = element_line(color = "black", size = 0.4),
    axis.text = element_text(color = "black"),
    legend.position = "none",  # or "top" with small text if needed
    plot.margin = margin(5, 5, 5, 5))+ labs(x = "Tank", y = expression("Eccentricity"), color = "Tank")

p.complexity <- scaled_df %>%
  filter(s.area_cm2 < 100) %>%
  ggplot(aes(prefix, complexity)) +
  geom_boxplot(aes(fill=batch),alpha=.7) +
  geom_jitter(width = .1,shape=21, aes(fill=batch))+
  scale_fill_manual(values = c("firebrick", "grey")) +
  theme_classic(base_size = 10) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(color = "black", size = 10, face = "plain"),
    axis.text.x = element_text(angle=45,hjust=1),
    axis.line = element_line(color = "black", size = 0.4),
    axis.ticks = element_line(color = "black", size = 0.4),
    axis.text = element_text(color = "black"),
    legend.position = "none",  # or "top" with small text if needed
    plot.margin = margin(5, 5, 5, 5))+ labs(x = "Tank", y = expression("Complexity"), color = "Tank")

p.combined <- (p.ecc + p.complexity + p.area.perimeter) +
  patchwork::plot_layout(widths = c(1, 1, 2))

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_ecc_comp_',format(Sys.Date(), "%m.%d.%y"),'.png'),
                plot= p.combined,
                height=3.5, width=8, dpi=300)

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_ecc_comp_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.combined,
                height=3.5, width=8)


p.combined.3 <- (p.area + p.perimeter + p.ecc + p.complexity + p.area.perimeter) +
  patchwork::plot_layout(widths = c(1, 1,1,1, 2))


ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_metrics_',format(Sys.Date(), "%m.%d.%y"),'.png'),
                plot= p.combined.3,
                height=3.5, width=10, dpi=300)




# prepare mask data
mask_df <- as.data.frame(which(labeled_2d > 0, arr.ind = TRUE))
colnames(mask_df) <- c("row", "col")
mask_df$label <- labeled_2d[cbind(mask_df$row, mask_df$col)]

# Merge with features (to access area)
mask_df <- left_join(mask_df, features, by = c("label" = "label"))
mask_df <- mask_df %>% filter(m.eccentricity != max(m.eccentricity))

mask_df %>% ggplot2::ggplot(ggplot2::aes(x = row , y = -col, fill = complexity)) +
  ggplot2::geom_tile() +
  ggplot2::scale_fill_viridis_c() +
  ggplot2::coord_fixed() +
  ggplot2::theme_void()

n_objects = length(unique(mask_df$label))
getPalette = colorRampPalette(RColorBrewer::brewer.pal(9, "Set1"))


p.masked <- mask_df %>%
  ggplot2::ggplot(ggplot2::aes(x = row , y = -col, fill = as.factor(label))) +
  ggplot2::geom_tile() +
  ggplot2::scale_fill_manual(values = getPalette(n_objects)) +
  ggplot2::coord_fixed() +
  ggplot2::theme_void() +
  ggplot2::theme(legend.position = "none")

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/masked_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.masked,
                height=3.5, width=8)
ggplot2::ggsave(filename=paste0('~/DATA/fabrics/masked_',format(Sys.Date(), "%m.%d.%y"),'.png'),
                plot= p.masked,
                height=4, width=4,dpi=300)


# Extract contours from the labeled mask
contours <- ocontour(labeled_2d)

# Convert contours into a data frame
contour_df <- do.call(rbind, lapply(seq_along(contours), function(i) {
  cbind(as.data.frame(contours[[i]]), label = i)
}))
colnames(contour_df) <- c('x', 'y', 'label')

# Plot only the outlines using ggplot2
p.contour <- ggplot(contour_df, aes(x = x, y = -y, group = label, color = as.factor(label))) +
  geom_path(linewidth = 0.5) +
  ggplot2::scale_color_manual(values = getPalette(n_objects+1)) +
  coord_fixed() +
  theme_void() +
  labs(color = "Object")

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/contours_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.contour,
                height=4, width=4)
ggplot2::ggsave(filename=paste0('~/DATA/fabrics/contours_',format(Sys.Date(), "%m.%d.%y"),'.png'),
                plot= p.contour,
                height=4, width=4,dpi=300)




######----- example image
p.contour <- contour_df %>%
  filter(label==2) %>%
  ggplot(aes(x = x, y = -y, group = label, color = as.factor(label))) +
  geom_path(linewidth = 0.5, color='darkgrey') +
  ggplot2::scale_color_manual(values = getPalette(n_objects+1)) +
  coord_fixed() +
  theme_void() +
  labs(color = "Object")+
  ggplot2::theme(legend.position = "none")+
  ggplot2::ggtitle(label = 'perimeter')

p.area <- mask_df %>%
  filter(label==2) %>%
  ggplot2::ggplot(ggplot2::aes(x = row , y = -col)) +
  ggplot2::geom_tile(fill='darkgrey') +
  #ggplot2::scale_fill_manual(values = getPalette(n_objects)) +
  ggplot2::coord_fixed() +
  ggplot2::theme_void() +
  ggplot2::theme(legend.position = "none")+
  ggplot2::ggtitle(label = 'area')


p.combined <- (p.area + p.contour) +
  patchwork::plot_layout(widths = c(1, 1))


# Initialize vector to store convex areas
convex_areas <- numeric(length = max(labeled_2d))

for (i in 1:max(labeled_2d)) {
  mask <- labeled_2d == i
  coords <- which(mask, arr.ind = TRUE)

  if (nrow(coords) < 3) {
    convex_areas[i] <- NA
    next
  }

  # Get convex hull indices
  hull_indices <- chull(coords)
  hull_coords <- coords[hull_indices, ]

  # Compute polygon area of convex hull
  convex_areas[i] <- geometry::polyarea(hull_coords[, 2], hull_coords[, 1])
}

# Add to features
features$convex_area <- convex_areas
features$solidity <- features$s.area / features$convex_area

# === 6b. Filter features by area ===
features$area <- features$s.area  # For clarity





# Filter out small objects (area < 500)
#features <- features[features$area >= 500, ]
#rownames(features) <- paste0("Obj", 1:nrow(features))  # Renumber after filtering

# Keep only corresponding labels
#valid_labels <- features$label
#labeled_2d[!labeled_2d %in% valid_labels] <- 0
#labeled_2d <- bwlabel(labeled_2d)  # Relabel again to keep things clean

# === 7. Save labeled overlay ===
dir.create("output", showWarnings = FALSE)

jpeg("output/labeled_overlay.jpg", res = 150, width = 1024, height = 768)
EBImage::display(colorLabels(labeled_2d), method = "raster")
text(x = features$m.cx, y = features$m.cy,
     labels = rownames(features), col = "white", cex = 0.7)
dev.off()

# === 8. Painted object overlay ===
jpeg("output/painted_overlay.jpg", res = 150, width = 1024, height = 768)
painted <- paintObjects(labeled_2d, toRGB(img_2d), col = "green", thick = TRUE)
EBImage::display(painted, method = "raster")
text(x = features$m.cx, y = features$m.cy - 20,
     labels = rownames(features), col = "gray90", cex = 0.5)
dev.off()

# === 9. Export features and data ===
write.table(features, "output/features.tsv", sep = "\t", quote = FALSE, row.names = TRUE)
saveRDS(labeled_2d, "output/labeled.rds")
saveRDS(features, "output/features.rds")



n_objects <- max(labeled_2d)
#n_row <- ceiling(sqrt(n_objects))
#n_col <- ceiling(n_objects / n_row)

# Set up plotting area for grid
#par(mfrow = c(n_row, n_col), mar = c(1,1,1,1))

#for(i in 1:n_objects) {
#  obj_mask <- labeled_2d == i
#  EBImage::display(obj_mask, method = "raster", all = FALSE, main = paste("Obj", i))
#}

par(mfrow = c(1,1))



getBoundingBox <- function(mask) {
  coords <- which(mask == 1, arr.ind = TRUE)
  y_min <- min(coords[,1])  # row min (vertical)
  y_max <- max(coords[,1])
  x_min <- min(coords[,2])  # col min (horizontal)
  x_max <- max(coords[,2])
  list(x = x_min, y = y_min, w = x_max - x_min + 1, h = y_max - y_min + 1)
}

# Manual crop function for EBImage Image objects
crop_image <- function(img, x, y, w, h) {
  img[y:(y + h - 1), x:(x + w - 1)]
}

crop_with_border <- function(img, x, y, w, h, border = 5) {
  # Calculate new coordinates with border
  x_new <- max(x - border, 1)
  y_new <- max(y - border, 1)

  x_max <- min(x + w - 1 + border, dim(img)[2])
  y_max <- min(y + h - 1 + border, dim(img)[1])

  w_new <- x_max - x_new + 1
  h_new <- y_max - y_new + 1

  img[y_new:(y_new + h_new - 1), x_new:(x_new + w_new - 1)]
}

# Input: labeled_2d is your labeled mask
n_objects <- max(labeled_2d)

# Prepare output folders
dir.create("output/object_masks", showWarnings = FALSE)

# Prepare a list to store cropped masks for tiling later
cropped_list <- list()
for(i in 1:n_objects) {
  obj_mask <- labeled_2d == i
  if (sum(obj_mask) == 0) next

  bbox <- getBoundingBox(obj_mask)
  cropped_mask <- crop_with_border(obj_mask, bbox$x, bbox$y, bbox$w, bbox$h, border = 10)
  writeImage(cropped_mask, paste0("output/object_masks/object_", i, ".png"))
  cropped_list[[i]] <- cropped_mask
}


# Now create tiled image of all cropped masks:

# Determine grid size (square-like)
n_row <- ceiling(sqrt(n_objects))
n_col <- ceiling(n_objects / n_row)

# Get max width and height for padding
max_w <- max(sapply(cropped_list, ncol))
max_h <- max(sapply(cropped_list, nrow))

# Function to pad images to same size
pad_to_max <- function(img, max_w, max_h) {
  w <- ncol(img)
  h <- nrow(img)
  padded <- matrix(FALSE, nrow = max_h, ncol = max_w)
  padded[1:h, 1:w] <- img
  return(padded)
}

# Pad all to same dimensions
cropped_list_padded <- lapply(cropped_list, pad_to_max, max_w = max_w, max_h = max_h)

# Combine into tiled image
tiled_img <- NULL
idx <- 1
for(r in 1:n_row) {
  row_imgs <- list()
  for(c in 1:n_col) {
    if(idx <= length(cropped_list_padded)) {
      row_imgs[[c]] <- cropped_list_padded[[idx]]
    } else {
      # Empty slot: black image
      row_imgs[[c]] <- matrix(FALSE, nrow = max_h, ncol = max_w)
    }
    idx <- idx + 1
  }
  # Combine row horizontally
  row_combined <- do.call(cbind, row_imgs)

  if(is.null(tiled_img)) {
    tiled_img <- row_combined
  } else {
    # Stack vertically
    tiled_img <- rbind(tiled_img, row_combined)
  }
}

# Convert logical matrix to EBImage Image object for saving/displaying
tiled_img_eb <- Image(tiled_img, colormode = "Grayscale")

# Save tiled image
writeImage(tiled_img_eb, "output/object_masks_tiled.png")

# Display tiled image
EBImage::display(tiled_img_eb, method = "raster")
EBImage::display(tiled_img, method = "raster" )


# plot convex hulls
# Function to draw convex hull overlay on an object
draw_convex_hull <- function(labeled_img, label_id, img_base = NULL) {
  mask <- labeled_img == label_id
  coords <- which(mask, arr.ind = TRUE)

  if (nrow(coords) < 3) return(NULL)  # Skip tiny objects

  ch <- chull(coords)  # Compute convex hull
  ch_coords <- coords[c(ch, ch[1]), ]  # Close polygon

  # Create an RGB base image for plotting
  if (is.null(img_base)) {
    img_base <- toRGB(normalize(mask))
  }

  # Draw polygon on the image
  img_hull <- drawPolygon(img_base,
                          x = ch_coords[,2],
                          y = ch_coords[,1],
                          col = "#FF4500", lwd = 2)

  return(img_hull)
}

# Get unique labels (excluding 0)
labels <- setdiff(unique(as.vector(labeled_2d)), 0)

# List to store hull-overlay images
hull_images <- list()

for (lbl in labels) {
  img_with_hull <- draw_convex_hull(labeled_2d, lbl)
  if (!is.null(img_with_hull)) {
    hull_images[[paste0("Obj", lbl)]] <- img_with_hull
  }
}




dir.create("output/hulls", showWarnings = FALSE, recursive = TRUE)

# Get object labels (non-zero)
labels <- sort(unique(as.vector(labeled_2d)))
labels <- labels[labels != 0]

# Calculate max width and height from bounding boxes (reuse your existing function)
get_bbox_dims <- function(mask) {
  coords <- which(mask == 1, arr.ind = TRUE)
  x_range <- range(coords[,2])
  y_range <- range(coords[,1])
  width <- diff(x_range) + 50
  height <- diff(y_range) + 50
  c(width = width, height = height)
}

bbox_dims <- do.call(rbind, lapply(labels, function(lbl) {
  mask <- labeled_2d == lbl
  get_bbox_dims(mask)
}))

max_width <- max(bbox_dims[, "width"])
max_height <- max(bbox_dims[, "height"])

# Set scale bar length in pixels (say 20 pixels)
scalebar_length <- 20

# Prepare plots
plot_list <- list()

for (lbl in labels) {
  mask <- labeled_2d == lbl
  coords <- which(mask == 1, arr.ind = TRUE)
  if (nrow(coords) < 3) next

  df <- as.data.frame(coords)
  colnames(df) <- c("y", "x")

  # Center coordinates
  cx <- mean(df$x)
  cy <- mean(df$y)
  df$x_centered <- df$x - cx
  df$y_centered <- - (df$y - cy)  # flip y for image coords

  hull_idx <- chull(df$x_centered, df$y_centered)
  hull_df <- df[hull_idx, ]

  # Axis limits for consistent scaling
  xlim <- c(-max_width / 2, max_width / 2)
  ylim <- c(-max_height / 2, max_height / 2)

  # Get solidity and area for this object (match label)
  sol <- features$solidity[features$label == lbl]
  area <- features$area[features$label == lbl]

  p <- ggplot(df, aes(x = x_centered, y = y_centered)) +
    geom_tile(fill = "gray20") +
    geom_polygon(data = hull_df, aes(x = x_centered, y = y_centered),
                 fill = NA, color = "red", linewidth = 0.6) +

    # Add scale bar at bottom left, offset inside plot limits
    # Add scale bar at bottom left, offset inside plot limits
    geom_segment(aes(x = xlim[1] + 5, y = ylim[1] + 5,
                     xend = xlim[1] + 5 + scalebar_length, yend = ylim[1] + 5),
                 color = "black", size = 1) +
    annotate("text", x = xlim[1] + 25 + scalebar_length / 2, y = ylim[1] + 30,
             label = paste0(scalebar_length, " px"), color = "black", size = 3, hjust = 0.5) +

    # Overlay solidity and area
    annotate("text", x = 0, y = ylim[2] - 50,
             label = sprintf("Solidity: %.3f\nArea: %d px", sol, round(area)),
             color = "orange", size = 3, hjust = 0.5, lineheight = 1) +

    coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE) +
    theme_void() +
    ggtitle(paste("Obj", lbl))

  plot_list[[as.character(lbl)]] <- p
}

# Combine plots in grid and save
grid_plot <- cowplot::plot_grid(plotlist = plot_list, ncol = 5)
ggsave("output/hulls_centered_with_scale_solidity.png", grid_plot, width = 15, height = 10, dpi = 150)



p.convex.hull <- ggplot(df, aes(x = x_centered, y = y_centered)) +
  geom_tile(fill = "lightgrey") +
  geom_polygon(data = hull_df, aes(x = x_centered, y = y_centered),
               fill = NA, color = "darkgrey", linewidth = 0.6, linetype='dashed') +
  annotate("text", x = xlim[1] + 25 + scalebar_length / 2, y = ylim[1] + 30,
           label = paste0(scalebar_length, " px"), color = "black", size = 3, hjust = 0.5) +
  # Overlay solidity and area
  #annotate("text", x = 0, y = ylim[2] - 50,
  #         label = sprintf("Solidity: %.3f\nArea: %d px", sol, round(area)),
  #         color = "orange", size = 3, hjust = 0.5, lineheight = 1) +
  coord_fixed(xlim = xlim, ylim = ylim/1.5, expand = FALSE) +
  theme_void() +
  ggtitle('Concex hull')



p.combined <- (p.area + p.contour + p.convex.hull) +
  patchwork::plot_layout(widths = c(1, 1,1))

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_example_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.combined,
                height=4, width=7)

#---------- Skeletonise

library(mmand)  # for thinning (skeletonization)
library(igraph) # for graph analysis

# Assume labeled_2d is your labeled mask (2D matrix)
labels <- sort(unique(as.vector(labeled_2d)))
labels <- labels[labels != 0]

skeleton_features <- data.frame(
  label = integer(),
  skeleton_length = numeric(),
  num_endpoints = integer(),
  num_branchpoints = integer()
)

for (lbl in labels) {
  mask <- labeled_2d == lbl
  mask <- as.array(mask)       # Convert to plain array (remove EBImage classes)
  mode(mask) <- "logical"      # Ensure logical mode
  skel <- skeletonise(mask, method = "hitormiss")  # or other method

  # Skeleton length = number of pixels in skeleton
  skel_length <- sum(skel)

  # Get coordinates of skeleton pixels
  coords <- which(skel == 1, arr.ind = TRUE)

  # Build graph from skeleton pixels where neighbors are connected
  # Define neighbors in 8 directions
  get_neighbors <- function(x, y) {
    expand.grid(
      x = (x-1):(x+1),
      y = (y-1):(y+1)
    ) %>%
      filter(!(x == !!x & y == !!y)) # exclude self
  }

  edges <- list()
  coord_to_id <- function(x, y) {
    paste(x, y, sep = "_")
  }

  # Map coords to IDs
  id_map <- apply(coords, 1, function(row) coord_to_id(row[1], row[2]))

  # Create graph edges by checking neighbors among skeleton pixels
  for (i in seq_len(nrow(coords))) {
    x <- coords[i,1]
    y <- coords[i,2]
    neighbors <- get_neighbors(x, y)

    # Filter neighbors inside skeleton
    neighbors_in_skel <- neighbors %>%
      filter(paste(x, y, sep = "_") %in% id_map)

    # Edges from current pixel to neighbors
    for (j in seq_len(nrow(neighbors_in_skel))) {
      nx <- neighbors_in_skel$x[j]
      ny <- neighbors_in_skel$y[j]
      if (coord_to_id(nx, ny) %in% id_map) {
        edges <- append(edges, list(c(coord_to_id(x,y), coord_to_id(nx, ny))))
      }
    }
  }

  # Build igraph object
  g <- graph_from_edgelist(do.call(rbind, edges), directed = FALSE)

  # Degree of each node (number of connections)
  degs <- degree(g)

  # Endpoints: degree == 1
  n_endpoints <- sum(degs == 1)

  # Branchpoints: degree > 2
  n_branchpoints <- sum(degs > 2)

  # Store results
  skeleton_features <- rbind(
    skeleton_features,
    data.frame(
      label = lbl,
      skeleton_length = skel_length,
      num_endpoints = n_endpoints,
      num_branchpoints = n_branchpoints
    )
  )

  # Optional: visualize skeleton overlay
  # display(combine(mask*0.5, skel), method = "raster")
}

# Join skeleton features to your main features data frame by label
features <- merge(features, skeleton_features, by.x = "label", by.y = "label", all.x = TRUE)







library(nat)

# Perform Sholl analysis on a skeleton image
sholl_analysis <- function(skel_matrix, center = NULL, step = 10, max_radius = NULL) {
  # Get all skeleton coordinates
  coords <- which(skel_matrix > 0, arr.ind = TRUE)  # row = y, col = x

  if (nrow(coords) == 0) stop("No skeleton pixels found.")

  # Define center
  if (is.null(center)) {
    center <- colMeans(coords)  # mean x and y
  }

  # Compute distances from center
  dists <- sqrt((coords[,1] - center[1])^2 + (coords[,2] - center[2])^2)

  # Define radii
  if (is.null(max_radius)) {
    max_radius <- ceiling(max(dists))
  }
  radii <- seq(step, max_radius, by = step)

  # Count intersections (points within a thin ring)
  counts <- sapply(radii, function(r) {
    sum(abs(dists - r) <= step / 2)
  })

  # Return as data.frame
  data.frame(radius = radii, intersections = counts)
}


# Assuming 'skel' is your skeleton matrix
result <- sholl_analysis(skel, step = 10)

# Plot
plot(result$radius, result$intersections, type = "b", col = "darkgreen",
     xlab = "Distance from center", ylab = "Number of intersections",
     main = "Sholl Analysis of Seaweed Skeleton")

plot(result$radius, result$intersections, type = "b", col = "darkgreen",
     xlab = "Distance from center", ylab = "Number of intersections",
     main = "Sholl Analysis")

#--------------
prune_skeleton <- function(skel, min_length = 20) {
  bin_skel <- skel > 0
  labeled <- bwlabel(bin_skel)

  keep <- logical(max(labeled))

  for (i in seq_len(max(labeled))) {
    coords <- which(labeled == i, arr.ind = TRUE)
    if (nrow(coords) >= min_length) {
      keep[i] <- TRUE
    }
  }

  pruned <- matrix(0, nrow = nrow(skel), ncol = ncol(skel))
  for (i in which(keep)) {
    pruned[labeled == i] <- 1
  }
  return(pruned)
}





pruned_skel <- prune_skeleton(skel, min_length = 3)
center <- find_skeleton_center_pruned(skel, min_branch_length = 3)
plot_sholl_overlay_with_mask(mask, pruned_skel, center)

#---------
get_user_center <- function(mask, skeleton) {
  df_mask <- as.data.frame(which(mask, arr.ind = TRUE))
  df_skel <- as.data.frame(which(skeleton > 0, arr.ind = TRUE))

  # Plot with flipped y for visual consistency
  plot(-df_mask$row ~ df_mask$col, pch = ".", col = "lightgreen", asp = 1, main = "Click to set center")
  points(-df_skel$row ~ df_skel$col, pch = ".", col = "black")

  message("Click once to select center...")
  center <- locator(1)
  return(c(center$y * -1, center$x))  # Return as [row, col]
}

run_sholl_analysis <- function(skeleton, center, step = 10, max_radius = NULL) {
  skel_coords <- which(skeleton > 0, arr.ind = TRUE)

  dists <- sqrt((skel_coords[,1] - center[1])^2 + (skel_coords[,2] - center[2])^2)
  if (is.null(max_radius)) max_radius <- floor(max(dists))

  radii <- seq(0, max_radius, by = step)

  # Intersections: close to exact radius
  intersections <- sapply(radii, function(r) {
    sum(abs(dists - r) < (step / 2))
  })

  # Length: total skeleton length in the shell [r - step/2, r + step/2)
  lengths <- sapply(radii, function(r) {
    sum(dists >= (r - step / 2) & dists < (r + step / 2))
  })

  list(
    radii = radii,
    intersections = intersections,
    lengths = lengths,
    skel_coords = skel_coords,
    center = center
  )
}

plot_sholl_overlay <- function(mask, skeleton, result, step = 10) {
  df_mask <- as.data.frame(which(mask, arr.ind = TRUE))
  df_skel <- as.data.frame(result$skel_coords)
  center <- result$center
  radii <- result$radii

  ggplot() +
    geom_tile(data = df_mask, aes(x = col, y = -row), fill = "lightgreen", alpha = 0.4) +
    geom_point(data = df_skel, aes(x = col, y = -row), color = "black", size = 0.3) +
    lapply(radii, function(r) {
      circle_df <- data.frame(
        angle = seq(0, 2*pi, length.out = 360)
      )
      circle_df$x <- center[2] + r * cos(circle_df$angle)
      circle_df$y <- -center[1] - r * sin(circle_df$angle)
      geom_path(data = circle_df, aes(x = x, y = y), color = "red", alpha = 0.4)
    }) +
    geom_point(aes(x = center[2], y = -center[1]), color = "blue", size = 3) +
    coord_fixed() +
    theme_void() +
    ggtitle("Skeleton Sholl")
}

plot_sholl_overlay2 <- function(mask, skeleton, result, step = 10) {
  df_mask <- as.data.frame(which(mask, arr.ind = TRUE))
  df_skel <- as.data.frame(result$skel_coords)
  center <- result$center
  radii <- result$radii

  ggplot() +
    lapply(radii, function(r) {
      circle_df <- data.frame(
        angle = seq(0, 2*pi, length.out = 360)
      )
      circle_df$x <- center[2] + r * cos(circle_df$angle)
      circle_df$y <- -center[1] - r * sin(circle_df$angle)
      geom_path(data = circle_df, aes(x = x, y = y), color = "grey", linetype='dashed')
    }) +
    geom_tile(data = df_mask, aes(x = col, y = -row), fill = "darkgrey", alpha = 0.4) +
    geom_point(data = df_skel, aes(x = col, y = -row), color = "darkgrey", size = 0.1) +
    #geom_point(aes(x = center[2], y = -center[1]), color = "darkgrey", size = 1) +
    coord_fixed() +
    theme_void() +
    ggtitle("Skeleton Sholl")
}


#set mask and skel?!
lbl = 2
mask <- labeled_2d == lbl
mask <- as.array(mask)       # Convert to plain array (remove EBImage classes)
mode(mask) <- "logical"      # Ensure logical mode
skel <- skeletonise(mask, method = "hitormiss")  # or other method


#get center from click
center <- get_user_center(mask, skel)

#run Sholl analysis
result <- run_sholl_analysis(skel, center, step = 20)

p.sholl <- plot_sholl_overlay(mask, skel, result)


ggplot2::ggsave(filename=paste0('~/DATA/fabrics/sholl_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.sholl,
                height=4, width=4)
ggplot2::ggsave(filename=paste0('~/DATA/fabrics/sholl_',format(Sys.Date(), "%m.%d.%y"),'.png'),
                plot= p.sholl,
                height=4, width=4,dpi=300)

p.sholl2 <- plot_sholl_overlay2(mask, skel, result)


p.combined <- (p.area + p.contour + p.convex.hull + p.sholl2) +
  patchwork::plot_layout(widths = c(1, 1,1,1))

ggplot2::ggsave(filename=paste0('~/DATA/fabrics/combined_morphometrics_',format(Sys.Date(), "%m.%d.%y"),'.pdf'),
                plot= p.combined,
                height=7, width=15)

# calculate sholl featues
sholl_features <- function(result) {
  df <- data.frame(radius = result$radii, intersections = result$intersections)

  peak_val <- max(df$intersections)
  peak_radius <- df$radius[which.max(df$intersections)]
  auc <- sum(df$intersections) * mean(diff(df$radius))
  slope <- coef(lm(intersections ~ radius, data = df))[2]

  list(
    peak_intersections = peak_val,
    peak_radius = peak_radius,
    auc = auc,
    slope = slope
  )
}

sholl_features(result)

#----

run_sholl_analysis <- function(skeleton, center, step = 10, max_radius = NULL) {
  skel_coords <- which(skeleton > 0, arr.ind = TRUE)
  dists <- sqrt((skel_coords[,1] - center[1])^2 + (skel_coords[,2] - center[2])^2)

  if (is.null(max_radius)) {
    max_radius <- floor(max(dists))
  }

  radii <- seq(0, max_radius, by = step)

  # Preallocate
  intersections <- numeric(length(radii))
  lengths <- numeric(length(radii))

  # Neighborhood offsets for 8-connectivity
  offsets <- expand.grid(dr = -1:1, dc = -1:1)
  offsets <- offsets[!(offsets$dr == 0 & offsets$dc == 0), ]

  # Loop over shells
  for (i in seq_along(radii)) {
    r <- radii[i]
    r_min <- r - step / 2
    r_max <- r + step / 2

    # Skeleton points in current shell
    in_shell <- which(dists >= r_min & dists < r_max)
    shell_coords <- skel_coords[in_shell, , drop = FALSE]

    # Count how many points are near this radius (for intersections)
    intersections[i] <- nrow(shell_coords)

    # Estimate length by checking connected neighbors
    length_px <- 0
    for (j in seq_len(nrow(shell_coords))) {
      pt <- shell_coords[j, ]
      count <- 0
      for (k in 1:nrow(offsets)) {
        nr <- pt[1] + offsets$dr[k]
        nc <- pt[2] + offsets$dc[k]
        if (nr > 0 && nc > 0 &&
            nr <= nrow(skeleton) && nc <= ncol(skeleton) &&
            skeleton[nr, nc] > 0) {
          count <- count + 1
        }
      }
      # Each connection is about 1 unit; divide by 2 to avoid double-counting
      length_px <- length_px + count / 2
    }
    lengths[i] <- length_px
  }

  list(
    radii = radii,
    intersections = intersections,
    lengths = lengths,
    skel_coords = skel_coords,
    center = center
  )
}

result <- run_sholl_analysis(skeleton, center = c(800, 600), step = 10)


#------------------------------------#


features %>% ggplot2::ggplot(ggplot2::aes(m.theta,b.mean))+
  ggplot2::geom_point()+
  ggplot2::geom_smooth(method='lm')+
  ggplot2::theme(aspect.ratio=1)


features %>% ggplot2::ggplot(ggplot2::aes(s.area,s.perimeter))+
  ggplot2::geom_point()+
  ggplot2::geom_smooth(method='lm')+
  ggplot2::theme(aspect.ratio=1)
