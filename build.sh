set -e  # this guy is a natural ctrl c :D

SRC_DIR=src 
INCLUDE_DIR=include
BUILD_DIR=build
MAIN_FILE=main.c
CC=gcc
CFLAGS="-I$INCLUDE_DIR -Wall -O2"

LIB_NAME=ViT

mkdir -p $BUILD_DIR

echo "=== Compiling library files ==="
LIB_OBJS=()

count=0
for src_file in $SRC_DIR/*.c; do
    obj_file="$BUILD_DIR/$(basename ${src_file%.c}.o)"
    echo "    $((++count)) Compiling $src_file -> $obj_file"
    $CC $CFLAGS -c "$src_file" -o "$obj_file"
    LIB_OBJS+=("$obj_file")

done

echo "    Creating static library lib$LIB_NAME.a"
ar rcs "$BUILD_DIR/lib$LIB_NAME.a" "${LIB_OBJS[@]}"

echo "=== Compiling main program ==="
MAIN_OBJ="$BUILD_DIR/main.o"
$CC $CFLAGS -c "$MAIN_FILE" -o "$MAIN_OBJ"

echo "=== Linking executable ==="
$CC "$MAIN_OBJ" -L$BUILD_DIR -l$LIB_NAME -o main -lm
$CC "$MAIN_OBJ" -L$BUILD_DIR -l$LIB_NAME -o main_debug -lm

echo "Build complete! Executable: main"
