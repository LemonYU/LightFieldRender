#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include"LightField.h"

int main()
{
	std::string imPath = "C:\\Users\\Yu\\Documents\\Nutstore\\Courses_code_win\\Computer Photography\\datasets\\toyLF\\";
	readImage(imPath);
	initWindow("toyLF");
	
	return 0;
}