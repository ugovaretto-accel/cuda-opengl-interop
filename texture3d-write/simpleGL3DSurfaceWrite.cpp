//Modified version of CUDA example provided by NVIDIA


/* 
    This example attempts to demonstrates how to write to 3D OpenGL textures
    using CUDA surfaces

	The steps are:
	1. Create an OpenGL 3D Texture
	2. Register the texture with CUDA
	3. Map the texture resource
	4. Get cudaArray pointer from the resource
	5. Pass the cudaArray to the device
	6, Bind the cudaArray to a globally scoped CUDA surface
	7. Call a CUDA kernel which writes 1.0f to the surface using surf3Dwrite
	8. Unmap the texture resource
	9. Get the data from the texture and validate it

    Host code
*/

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif


// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// includes, system
#include <iostream>

using namespace std;

#define REFRESH_DELAY	  10 //ms


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;


GLuint texID;
cudaGraphicsResource *cuda_image_resource;
cudaArray            *cuda_image_array;

dim3 textureDim(128, 128, 128);


extern "C" 
void launch_kernel(struct cudaArray *cuda_image_array, dim3 texture_dim);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward


bool initGL(int *argc, char** argv);
void initCUDA();

void checkTex();

void runCudaTest();

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void timerEvent(int value);

void CHECK_CUDA(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}  

void CHECK_ERROR_GL() {
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        std::cerr << "GL Error: " << gluErrorString(err) << std::endl;
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	initGL(&argc, argv);
	initCUDA();

	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_3D, texID);
	{
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);

		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, textureDim.x, textureDim.y, textureDim.z, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_3D, 0);

	CHECK_ERROR_GL();

	// register Image (texture) to CUDA Resource
	CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_image_resource, 
                                           texID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

	// map CUDA resource
	CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_image_resource, 0));
	{
		//Get mapped array
		CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_image_array, cuda_image_resource, 0, 0));
		launch_kernel(cuda_image_array, textureDim);
	}
	CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_image_resource, 0));

	checkTex();

	CHECK_CUDA(cudaGraphicsUnregisterResource(cuda_image_resource));

	glDeleteTextures(1, &texID);
	
	CHECK_CUDA(cudaDeviceReset());
}


////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA GL 3D Texture Surface Write");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 "))
	{
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CHECK_ERROR_GL();

    return true;
}

void initCUDA()
{
	int deviceCount;
	CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

	cerr << "CUDA device count: " << deviceCount << endl;
	int device = 0; //SELECT GPU HERE
	cerr << "Selecting GPU: " << device << endl;
	CHECK_CUDA(cudaSetDevice(device));
	CHECK_CUDA(cudaGLSetGLDevice( device ));
}


////////////////////////////////////////////////////////////////////////////////
//! Run the CUDA test
////////////////////////////////////////////////////////////////////////////////
void runCudaTest()
{
	// map CUDA resource
	CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_image_resource, 0));
	{
		//Get mapped array
		CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_image_array, cuda_image_resource, 0, 0));
		launch_kernel(cuda_image_array, textureDim);
	}
	CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_image_resource, 0));
}


////////////////////////////////////////////////////////////////////////////////
//! Check Texture
////////////////////////////////////////////////////////////////////////////////
void checkTex()
{
	int numElements = textureDim.x*textureDim.y*textureDim.z*4;
	float *data = new float[numElements];

	glBindTexture(GL_TEXTURE_3D, texID);
	{
		glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, data);
	}
	glBindTexture(GL_TEXTURE_3D, 0);

	bool fail = false;
	for(int i = 0; i < numElements && !fail; i++)
	{
		if(data[i] != 1.0f)
		{
			cerr << "Not 1.0f, failed writing to texture" << endl;
			fail = true;
		}
	}
	if(!fail)
	{
		cerr << "All Elements == 1.0f, texture write successful" << endl;
	}

	delete [] data;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutSwapBuffers();
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        exit(0);
        break;
    }
}

